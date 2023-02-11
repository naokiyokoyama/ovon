import inspect
import random
from dataclasses import dataclass

import gym.spaces
import numpy as np
import torch
from gym import spaces
from habitat import logger
from habitat.config import read_write
from habitat_baselines import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    rank0_only,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.utils.common import (
    batch_obs,
    get_num_actions,
    is_continuous_action_space,
    inference_mode,
)
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import optim

from distributed_dagger.distributed_dagger.dagger import DAggerDDP
from ovon.models.model_utils import ComputeLogProbsMixin, RecurrentPolicyMixin

VISUAL_FEATURES_KEY = PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY


@baseline_registry.register_trainer(name="dagger")
class DAggerTrainer(DAggerDDP, BaseTrainer):
    actor_critic = NetPolicy
    continuous_actions: bool = False
    obs_transforms = []
    observation_space: spaces.Dict
    prev_actions: torch.tensor
    policy_action_space: gym.spaces.Space
    teacher = NetPolicy

    def __init__(self, config):
        self.config = config
        self._is_distributed = get_distrib_size()[2] > 1
        dagger_cfg = config.habitat_baselines.dagger

        logger.add_filehandler(config.habitat_baselines.log_file)
        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        self.distribute_gpus()
        self.device = torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(self.device)

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=False,
        )
        self.masks = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        self.static_encoder = not config.habitat_baselines.rl.ddppo.train_encoder
        self.setup_obs_action_space()
        self.teacher = self.setup_policy(dagger_cfg.teacher_policy.name)
        self.actor_critic = self.setup_policy(dagger_cfg.policy.name, lp_mixin=True)
        if self.static_encoder:
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        optimizer = self.get_optimizer()

        super().__init__(
            envs=self.envs,
            actor_critic=self.actor_critic,
            optimizer=optimizer,
            batch_length=dagger_cfg.batch_length,
            total_num_steps=dagger_cfg.total_num_steps,
            device=self.device,
            updates_per_ckpt=dagger_cfg.updates_per_ckpt,
            num_updates=dagger_cfg.num_updates,
            teacher_forcing=dagger_cfg.teacher_forcing,
            tb_dir=config.habitat_baselines.tensorboard_dir,
            checkpoint_folder=config.habitat_baselines.checkpoint_folder,
        )

    def setup_obs_action_space(self):
        self.obs_transforms = get_active_obs_transforms(self.config)
        self.observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], self.obs_transforms
        )

        self.policy_action_space = self.envs.action_spaces[0]
        self.continuous_actions = is_continuous_action_space(self.policy_action_space)
        if self.continuous_actions:
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(self.policy_action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True
        self.prev_actions = torch.zeros(
            self.num_envs,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )

    def setup_policy(self, policy_name, lp_mixin=False):
        # fmt: off
        policy_cls = baseline_registry.get_policy(policy_name)
        classes = [ComputeLogProbsMixin, policy_cls] if lp_mixin else [policy_cls]
        class MixedPolicy(RecurrentPolicyMixin, *classes): pass  # noqa
        # fmt: on
        policy = MixedPolicy.from_config(
            self.config, self.observation_space, self.policy_action_space
        )
        policy.init_hidden_states(self.num_envs, self.device)
        policy.to(self.device)
        return policy

    def transform_observations(self, observations):
        """Applied to the observations output of self.envs.step()"""
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return batch

    def sift_env_outputs(self, outputs):
        """Applied to all outputs of self.envs.step()"""
        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
        rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        self.masks = torch.logical_not(dones)
        return observations, rewards, dones, infos

    def get_teacher_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.action_loss()"""
        with inference_mode():
            action = self.teacher.act(observations, self.prev_actions, self.masks)[1]
        return action.clone()

    def get_student_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.envs.step()"""
        # Must clone observations to avoid inference tensor error
        obs_clone = TensorDict({k: v.clone() for k, v in observations.items()})
        if self.static_encoder:
            with inference_mode():
                visual_feats = self.actor_critic.net.visual_encoder(observations)
            obs_clone[VISUAL_FEATURES_KEY] = visual_feats.clone()
        _, actions, _ = self.actor_critic.act(obs_clone, self.prev_actions, self.masks)
        self.prev_actions.copy_(actions)  # noqa
        return actions

    def step_envs(self, actions):
        if self.continuous_actions:
            l, h = self.policy_action_space.low, self.policy_action_space.high  # noqa
            actions = [np.clip(a.numpy(), l, h) for a in actions.cpu()]
        actions = [a.item() for a in actions.cpu()]
        return super().step_envs(actions)

    def update_tensorboard(self, observations, rewards, dones, infos, action_loss):
        pass

    def distribute_gpus(self):
        if not self._is_distributed:
            return
        local_rank, tcp_store = init_distrib_slurm(
            self.config.habitat_baselines.rl.ddppo.distrib_backend
        )
        if rank0_only():
            logger.info(
                f"Initialized DAgger with {torch.distributed.get_world_size()} workers"  # noqa
            )

        with read_write(self.config):
            self.config.habitat_baselines.torch_gpu_id = local_rank
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
            # Multiply by the number of simulators to make sure they also get unique
            # seeds
            self.config.habitat.seed += (
                torch.distributed.get_rank()  # noqa
                * self.config.habitat_baselines.num_environments
            )

        random.seed(self.config.habitat.seed)
        np.random.seed(self.config.habitat.seed)
        torch.manual_seed(self.config.habitat.seed)

    def get_optimizer(self):
        optim_cls = optim.Adam
        params = list(filter(lambda p: p.requires_grad, self.actor_critic.parameters()))
        dagger_cfg = self.config.habitat_baselines.dagger
        optim_kwargs = dict(params=params, lr=dagger_cfg.lr, eps=dagger_cfg.eps)
        signature = inspect.signature(optim_cls.__init__)
        if "foreach" in signature.parameters:
            optim_kwargs["foreach"] = True
        else:
            try:
                import torch.optim._multi_tensor  # noqa

                optim_cls = torch.optim._multi_tensor.Adam  # noqa
            except ImportError:
                pass

        optimizer = optim_cls(**optim_kwargs)
        return optimizer


@dataclass
class DAggerConfig(HabitatBaselinesBaseConfig):
    batch_length: int = 20
    lr: float = 3e-4
    eps: float = 1e-5
    num_updates: int = 100
    policy: PolicyConfig = PolicyConfig()
    teacher_forcing: bool = False
    teacher_policy: PolicyConfig = PolicyConfig()
    total_num_steps: float = 1e6
    updates_per_ckpt: int = 100


@dataclass
class HabitatBaselinesDAggerConfig(HabitatBaselinesRLConfig):
    # We still need to inherit from RLConfig to get support for preemption...
    dagger: DAggerConfig = DAggerConfig()


cs = ConfigStore.instance()
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_dagger_config_base",
    node=HabitatBaselinesDAggerConfig(),
)
