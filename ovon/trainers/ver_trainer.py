import time
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn
from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines import VERTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch, apply_obs_transforms_obs_space,
    get_active_obs_transforms)
from habitat_baselines.rl.ddppo.ddp_utils import (EXIT, load_resume_state,
                                                  rank0_only, requeue_job,
                                                  save_resume_state)
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.timing import Timing
from habitat_baselines.utils.common import cosine_decay, inference_mode
from torch.optim.lr_scheduler import LambdaLR

from ovon.algos.ppo import DDPPO, PPO
from ovon.utils.lr_scheduler import PIRLNavLRScheduler

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_trainer(name="ver_pirlnav")
class VERPIRLNavTrainer(VERTrainer):

    def _setup_actor_critic_agent(self, ppo_cfg: "DictConfig") -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self.config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self.config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self.config.habitat_baselines.rl.ddppo.pretrained:
            missing_keys = self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                },
                strict=False,
            )
            logger.info("Missing keys when loading pretrained wieghts: {}".format(missing_keys))
        elif self.config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.habitat_baselines.rl.ddppo.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        policy_cfg = self.config.habitat_baselines.rl.policy
        if policy_cfg.finetune.enabled:
            self.actor_critic.freeze_visual_encoders()
            self.actor_critic.freeze_state_encoder()
            self.actor_critic.freeze_actor()

        if self.config.habitat_baselines.rl.ddppo.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO).from_config(
            self.actor_critic, ppo_cfg
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training VER using PIRLNav LR scheduler.

        Returns:
            None
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        self.num_steps_done = 0
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config = resume_state["config"]

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]

        self._init_train(resume_state)

        count_checkpoints = 0
        policy_cfg = self.config.habitat_baselines.rl.policy

        if policy_cfg.finetune.enabled:
            lr_scheduler = PIRLNavLRScheduler(
                optimizer=self.agent.optimizer,
                agent=self.agent,
                num_updates=self.config.habitat_baselines.num_updates,
                base_lr=self.config.habitat_baselines.rl.ppo.lr,
                finetuning_lr=policy_cfg.finetune.lr,
                ppo_eps=self.config.habitat_baselines.rl.ppo.eps,
                start_actor_update_at=policy_cfg.finetune.start_actor_update_at,
                start_actor_warmup_at=policy_cfg.finetune.start_actor_warmup_at,
                start_critic_update_at=policy_cfg.finetune.start_critic_update_at,
                start_critic_warmup_at=policy_cfg.finetune.start_critic_warmup_at,
            )
        else:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: cosine_decay(self.percent_done()),
            )

        if resume_state is not None:
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if self.ver_config.overlap_rollouts_and_learn:
            self.preemption_decider.start_rollout()

        while not self.is_done():
            profiling_wrapper.on_start_step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - self.percent_done()
                )

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    report_worker_state=self.report_worker.state_dict(),
                )
                resume_state = dict(
                    state_dict=self.agent.state_dict(),
                    optim_state=self.agent.optimizer.state_dict(),
                    lr_sched_state=lr_scheduler.state_dict(),
                    config=self.config,
                    requeue_stats=requeue_stats,
                )

                save_resume_state(
                    resume_state,
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update
                [w.close() for w in self._all_workers]
                [w.join() for w in self._all_workers]

                requeue_job()
                break

            with inference_mode():
                if not self.ver_config.overlap_rollouts_and_learn:
                    self.preemption_decider.start_rollout()
                    while not self.rollouts.rollout_done:
                        self._inference_worker_impl.try_one_step()

                    self._inference_worker_impl.finish_rollout()

                self._iw_sync.rollout_done.wait()
                self._iw_sync.rollout_done.clear()

                if self._iw_sync.all_workers.n_waiting > 0:
                    raise RuntimeError(
                        f"{self._iw_sync.all_workers.n_waiting} inference worker(s) is(are) still waiting on the IW barrier. "
                        "Likely they never waited on it.\n"
                    )

                self.rollouts.after_rollout()

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.learning_rollouts.copy(self.rollouts)

                self.preemption_decider.end_rollout(
                    self.rollouts.num_steps_to_collect
                )

                self.queues.report.put(
                    (
                        ReportWorkerTasks.num_steps_collected,
                        int(self.rollouts.num_steps_collected),
                    )
                )

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.rollouts.after_update()
                        self._iw_sync.should_start_next.set()
                        self.preemption_decider.start_rollout()

            losses = self._update_agent()
            lrs = {}
            for i, param_group in enumerate(self.agent.optimizer.param_groups):
                lrs["lr_{}".format(i)] = param_group["lr"]
            
            learner_metrics = {
                **losses,
                **lrs,
            }

            self.preemption_decider.learner_time(self._learning_time)

            self.queues.report.put_many(
                (
                    (
                        ReportWorkerTasks.learner_timing,
                        self.timer,
                    ),
                    (
                        ReportWorkerTasks.learner_update,
                        learner_metrics,
                    ),
                )
            )
            self.timer = Timing()

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_steps_done = int(self.report_worker.num_steps_done)

            self.num_updates_done += 1
            # checkpoint model
            if rank0_only() and self.should_checkpoint():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1

        self.window_episode_stats = (
            self.report_worker.get_window_episode_stats()
        )

        [w.close() for w in self._all_workers]
        [w.join() for w in self._all_workers]

        if self._is_distributed:
            torch.distributed.barrier()
