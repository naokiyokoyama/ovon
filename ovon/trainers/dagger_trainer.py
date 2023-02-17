import inspect
import random
from dataclasses import dataclass
from typing import Tuple, Union

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from habitat import logger
from habitat.config import read_write
from habitat_baselines import BaseTrainer, RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.utils.common import (
    batch_obs,
    get_num_actions,
    is_continuous_action_space,
)
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import optim

from distributed_dagger.ddp import rank0_only
from distributed_dagger.distributed_dagger.dagger import DAggerDDP
from ovon.models.model_utils import DAggerPolicyMixin
from ovon.utils.rollout_storage_no_2d import RolloutStorageNo2D

VISUAL_FEATURES_KEY = PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY


@baseline_registry.register_trainer(name="dagger")
class DAggerTrainer(DAggerDDP, BaseTrainer):  # noqa
    actor_critic = NetPolicy
    action_shape: Tuple[int]
    continuous_actions: bool = False
    obs_transforms = []
    observation_space: spaces.Dict
    prev_actions: torch.tensor
    policy_action_space: gym.spaces.Space
    rollouts: Union[RolloutStorage, RolloutStorageNo2D]
    teacher = NetPolicy

    def __init__(self, config):
        self.config = config
        dagger_cfg = config.habitat_baselines.dagger

        self.distribute_gpus()
        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.add_filehandler(config.habitat_baselines.log_file)
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=False,
        )

        self.setup_obs_action_space()
        self.teacher = self.setup_policy(dagger_cfg.teacher_policy.name)
        self.actor_critic = self.setup_policy(dagger_cfg.policy.name)
        self.static_encoder = not config.habitat_baselines.rl.ddppo.train_encoder
        if self.static_encoder:
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        # Masks to support Habitat's recurrent policies
        self.masks = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        if self.is_distributed:
            self.init_distributed(find_unused_params=True)

        super().__init__(
            envs=self.envs,
            actor_critic=self.actor_critic,
            optimizer=self.get_optimizer(),
            batch_length=dagger_cfg.batch_length,
            total_num_steps=dagger_cfg.total_num_steps,
            device=self.device,
            updates_per_ckpt=dagger_cfg.updates_per_ckpt,
            num_updates=config.habitat_baselines.num_updates,
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
            self.action_shape = (get_num_actions(self.policy_action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            self.action_shape = (1,)
            discrete_actions = True
        self.prev_actions = torch.zeros(
            self.num_envs,
            *self.action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )

    def setup_policy(self, policy_name):
        # fmt: off
        policy_cls = baseline_registry.get_policy(policy_name)
        class MixedPolicy(DAggerPolicyMixin, policy_cls): pass  # noqa
        # fmt: on
        policy = MixedPolicy.from_config(
            self.config, self.observation_space, self.policy_action_space
        )
        policy.init_hidden_states(self.num_envs, self.device)
        policy.to(self.device)
        return policy

    def transform_observations(self, observations):
        """Applied to the observations output of self.sift_env_outputs()"""
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return batch

    def sift_env_outputs(self, outputs):
        """Applied to outputs of self.envs.step()"""
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        self.masks = torch.logical_not(dones).reshape(self.num_envs, 1)
        return observations, None, dones, infos  # Don't need rewards for DAgger

    def get_teacher_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.action_loss()"""
        actions = self.teacher.act(observations, self.prev_actions, self.masks)
        return actions

    def get_student_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.envs.step()"""
        actions = self.actor_critic.act(observations, self.prev_actions, self.masks)
        self.prev_actions.copy_(actions)  # noqa
        return actions

    def initialize_rollout(self, observations):
        ppo_cfg = self.config.habitat_baselines.rl.ppo
        rollout_kwargs = dict(
            numsteps=self.config.habitat_baselines.dagger.batch_length,
            num_envs=self.envs.num_envs,
            action_space=self.policy_action_space,
            recurrent_hidden_state_size=ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=False,
            action_shape=self.action_shape,
            discrete_actions=not self.continuous_actions,
        )
        if self.static_encoder:
            rollout_cls = RolloutStorageNo2D
            rollout_kwargs["initial_obs"] = observations
            rollout_kwargs["visual_encoder"] = self.actor_critic.net.visual_encoder
            rollout_kwargs["observation_space"] = spaces.Dict(
                {
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self.actor_critic.net.visual_encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **self.observation_space.spaces,
                }
            )
        else:
            rollout_cls = RolloutStorage
            rollout_kwargs["observation_space"] = self.observation_space
        self.rollouts = rollout_cls(**rollout_kwargs)
        self.rollouts.to(self.device)
        if not self.static_encoder:
            self.rollouts.buffers["observations"][0] = observations

    def update_rollout(self, observations, teacher_actions):
        self.rollouts.insert(
            next_observations=observations,
            next_recurrent_hidden_states=self.actor_critic.rnn_hidden_states,
            actions=teacher_actions,
            next_masks=self.masks,
        )
        self.rollouts.advance_rollout()

    def get_last_obs(self):
        last_buff = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]
        return last_buff["observations"]

    def update(self):
        data_generator = self.rollouts.recurrent_generator(
            advantages=None,
            num_mini_batch=self.config.habitat_baselines.rl.ppo.num_mini_batch,
        )
        losses = []
        for epoch in range(self.config.habitat_baselines.rl.ppo.ppo_epoch):
            for batch in data_generator:
                if self.config.habitat_baselines.dagger.loss_type == "log_prob":
                    log_probs = self._with_grad(
                        "evaluate_log_probs",
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                        batch["rnn_build_seq_info"],
                    )
                    loss = -log_probs.mean()
                elif self.config.habitat_baselines.dagger.loss_type == "mse":
                    # Assumes student has Categorical action distribution
                    logits = -self._with_grad(
                        "get_logits",
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["rnn_build_seq_info"],
                    )
                    N = batch["recurrent_hidden_states"].shape[0]
                    T = batch["actions"].shape[0] // N
                    actions_batch = batch["actions"].view(T, N, -1)
                    logits = logits.view(T, N, -1)
                    loss = F.cross_entropy(
                        logits.permute(0, 2, 1), actions_batch.squeeze(-1)
                    )
                else:
                    raise ValueError(
                        "Invalid loss type:"
                        f"{self.config.habitat_baselines.dagger.loss_type}"
                    )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        self.rollouts.after_update()
        mean_loss = np.mean(losses)
        return mean_loss

    def step_envs(self, actions):
        if self.continuous_actions:
            l, h = self.policy_action_space.low, self.policy_action_space.high  # noqa
            actions = [np.clip(a.numpy(), l, h) for a in actions.cpu()]
        actions = [a.item() for a in actions.cpu()]
        return super().step_envs(actions)

    def log_data(self, mean_loss):
        fps, mean_stats, stats_ordering = super().log_data(mean_loss)
        total_steps = self.num_steps_done * self.num_workers
        logger.info(
            f"update: {self.num_updates_done}\tfps: {fps:.3f}\tsteps: "
            f"{total_steps}\tavg loss: {mean_stats[-1]:.3f}"
        )
        window_scalars = []
        for idx, k in enumerate(stats_ordering):
            window_scalars.append(f"{k}: {mean_stats[idx]:.3f}")
        logger.info("  ".join(window_scalars))

    def distribute_gpus(self):
        if get_distrib_size()[2] > 1:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
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
            self.num_workers = torch.distributed.get_world_size()  # noqa
            if rank0_only():
                logger.info(f"Initialized DDP DAgger with {self.num_workers} workers")
        else:
            logger.info(f"Initialized non-distributed DAgger")

        self.device = torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)

        random.seed(self.config.habitat.seed)
        np.random.seed(self.config.habitat.seed)
        torch.manual_seed(self.config.habitat.seed)
        torch.cuda.set_device(self.device)

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
    loss_type: str = "log_prob"


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
