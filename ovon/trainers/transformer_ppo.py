#!/usr/bin/env python3


import collections
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import LagrangeInequalityCoefficient, inference_mode


class MinimalTransformerPPO(PPO):
    """
    Custom PPO implementation to handle the transformer-based policy window
    contexts.
    """

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
        skipgrad: bool = False,
        skipgrad_factor1: float = 0.1,
        skipgrad_factor2: int = 2,
        grad_accum_mini_batches: int = 1,
        optimizer_name: str = "adam",
        adamw_weight_decay: float = 0.01,
        ignore_old_obs_grad: bool = False,
    ) -> None:
        self.skipgrad = skipgrad
        self.skipgrad_factor1 = skipgrad_factor1
        self.skipgrad_factor2 = skipgrad_factor2
        self.grad_accum_mini_batches = grad_accum_mini_batches
        self.optimizer_name = optimizer_name
        self.adamw_weight_decay = adamw_weight_decay
        self.ignore_old_obs_grad = ignore_old_obs_grad

        super().__init__(
            actor_critic=actor_critic,
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_normalized_advantage=use_normalized_advantage,
            entropy_target_factor=entropy_target_factor,
            use_adaptive_entropy_pen=use_adaptive_entropy_pen,
        )

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        assert (
            config.num_mini_batch >= config.grad_accum_mini_batches
            and config.num_mini_batch % config.grad_accum_mini_batches == 0
        )
        obj = cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            skipgrad=config.skipgrad,
            skipgrad_factor1=config.skipgrad_factor1,
            skipgrad_factor2=config.skipgrad_factor2,
            grad_accum_mini_batches=config.grad_accum_mini_batches,
            optimizer_name=config.optimizer_name,
            adamw_weight_decay=config.adamw_weight_decay,
            ignore_old_obs_grad=config.ignore_old_obs_grad,
        )
        return obj

    def update(
        self, rollouts: RolloutStorage, loss_from_step: int = 0
    ) -> Dict[str, float]:
        advantages = self.get_advantages(rollouts)

        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.data_generator(advantages, self.num_mini_batch)

            self._update_from_data_generator(
                data_generator, epoch, rollouts, learner_metrics, loss_from_step
            )
            # for _bid, batch in enumerate(data_generator):
            #     self._update_from_batch(
            #         batch, epoch, rollouts, learner_metrics
            #     )

            profiling_wrapper.range_pop()  # PPO.update epoch

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    def _create_optimizer(self, lr, eps):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        logger.info(
            f"Number of params to train: {sum(param.numel() for param in params)}"
        )
        if len(params) > 0:
            if self.optimizer_name.lower() == "adamw":
                print("Creating AdamW optimizer")
                optim_cls = optim.AdamW
                _kwargs = {"weight_decay": self.adamw_weight_decay}
            else:
                optim_cls = optim.Adam
                _kwargs = {}
            optim_kwargs = dict(params=params, lr=lr, eps=eps, **_kwargs)
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            return optim_cls(**optim_kwargs)
        else:
            return None

    def _update_from_data_generator(
        self, data_generator, epoch, rollouts, learner_metrics, loss_from_step
    ):
        """
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()

        for _bid, batch in enumerate(data_generator):
            with torch.autocast("cuda"):
                n_envs, seq_len = batch["rnn_build_seq_info"]["dims"]
                # inv_weights = (~batch["masks"]).to(batch["advantages"].dtype).unflatten(0, (n_envs, seq_len))[:, loss_from_step:].sum(axis=-2, keepdim=True) + 1
                rnn_build_seq_info = batch.get("rnn_build_seq_info", None)
                if self.ignore_old_obs_grad:
                    rnn_build_seq_info["stop_grad_steps"] = torch.tensor(loss_from_step)
                inv_weights = 1.0
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_loss_res,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                    rnn_build_seq_info,
                )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])

                surr1 = batch["advantages"] * ratio
                surr2 = batch["advantages"] * (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                )
                action_loss = -torch.min(surr1, surr2)

                orig_values = values

                if self.use_clipped_value_loss:
                    delta = values.detach() - batch["value_preds"]
                    value_pred_clipped = batch["value_preds"] + delta.clamp(
                        -self.clip_param, self.clip_param
                    )

                    values = torch.where(
                        delta.abs() < self.clip_param,
                        values,
                        value_pred_clipped,
                    )

                value_loss = 0.5 * F.mse_loss(
                    values, batch["returns"], reduction="none"
                )

                if "is_coeffs" in batch:
                    assert isinstance(batch["is_coeffs"], torch.Tensor)
                    ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)

                    def mean_fn(t):
                        return torch.mean(
                            ver_is_coeffs
                            * t.unflatten(0, (n_envs, seq_len))[
                                :, loss_from_step:
                            ].flatten(0, 1)
                        )

                else:

                    def mean_fn(t):
                        return torch.mean(
                            (t.unflatten(0, (n_envs, seq_len)) / inv_weights)[
                                :, loss_from_step:
                            ].flatten(0, 1)
                        )

                action_loss, value_loss, dist_entropy = map(
                    mean_fn,
                    (action_loss, value_loss, dist_entropy),
                )

                all_losses = [
                    self.value_loss_coef * value_loss,
                    action_loss,
                ]

                if isinstance(self.entropy_coef, float):
                    all_losses.append(-self.entropy_coef * dist_entropy)
                else:
                    all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

                all_losses.extend(v["loss"] for v in aux_loss_res.values())

                total_loss = torch.stack(all_losses).sum()

                total_loss = self.before_backward(total_loss)

            total_loss.backward()
            self.after_backward(total_loss)

            if (_bid + 1) % self.grad_accum_mini_batches == 0:
                grad_norm = self.before_step()

                if self.skipgrad:
                    if not hasattr(self, "_grad_norm_ema"):
                        self._grad_norm_ema = grad_norm
                        self._updates_count_ema = 1
                        print("Creating grad_norm EMA stats.")
                    else:
                        _fac = self.skipgrad_factor1
                        SHOULD_UPDATE = grad_norm < 20000000 * self._grad_norm_ema
                        if SHOULD_UPDATE:
                            self._grad_norm_ema = (
                                self._grad_norm_ema * (1 - _fac) + _fac * grad_norm
                            )
                        else:
                            self._grad_norm_ema = (
                                self._grad_norm_ema * (1 - _fac)
                                + _fac * self.skipgrad_factor2 * self._grad_norm_ema
                            )

                    if self._updates_count_ema < 100 or SHOULD_UPDATE:
                        self._updates_count_ema += 1
                    else:
                        self.optimizer.zero_grad()
                        print(
                            f"Skipping step {self._updates_count_ema} because of high"
                            " norm."
                        )
                        # print(f"_grad_norm_ema={self._grad_norm_ema}, grad_norm={grad_norm}")
                self.optimizer.step()
                self.after_step()
                self._set_grads_to_none()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            record_min_mean_max(batch["returns"], "batch_returns")
            record_min_mean_max(batch["value_preds"], "batch_value_preds")
            record_min_mean_max(batch["advantages"], "batch_advantages")
            record_min_mean_max(batch["rewards"], "batch_rewards")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(self.entropy_coef().detach())

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (rollouts.current_policy_version - batch["policy_version"]).float(),
                    "policy_version_difference",
                )

    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()

        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch.get("rnn_build_seq_info", None),
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(values, batch["returns"], reduction="none")

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)

            def mean_fn(t):
                return torch.mean(ver_is_coeffs * t)

        else:
            mean_fn = torch.mean

        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()

        if self.skipgrad:
            if not hasattr(self, "_grad_norm_ema"):
                self._grad_norm_ema = grad_norm
                self._updates_count_ema = 1
                print("Creating grad_norm EMA stats.")
            else:
                _fac = self.skipgrad_factor1
                SHOULD_UPDATE = grad_norm < 20000000 * self._grad_norm_ema
                if SHOULD_UPDATE:
                    self._grad_norm_ema = (
                        self._grad_norm_ema * (1 - _fac) + _fac * grad_norm
                    )
                else:
                    self._grad_norm_ema = (
                        self._grad_norm_ema * (1 - _fac)
                        + _fac * self.skipgrad_factor2 * self._grad_norm_ema
                    )

            if self._updates_count_ema < 100 or SHOULD_UPDATE:
                self._updates_count_ema += 1
            else:
                self.optimizer.zero_grad()
                print(f"Skipping step {self._updates_count_ema} because of high norm.")
                # print(f"_grad_norm_ema={self._grad_norm_ema}, grad_norm={grad_norm}")
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            record_min_mean_max(batch["returns"], "batch_returns")
            record_min_mean_max(batch["value_preds"], "batch_value_preds")
            record_min_mean_max(batch["advantages"], "batch_advantages")
            record_min_mean_max(batch["rewards"], "batch_rewards")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(self.entropy_coef().detach())

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (rollouts.current_policy_version - batch["policy_version"]).float(),
                    "policy_version_difference",
                )


class DistributedMinimalTransformerPPO(
    DecentralizedDistributedMixin, MinimalTransformerPPO
):
    pass


class TransformerPPO(PPO):
    """
    Custom PPO implementation to handle the transformer-based policy window
    contexts.
    """

    def after_update(self) -> None:
        pass
        # print("Stepping LR scheduler")
        # self._lr_scheduler.step()

    def _create_optimizer(self, lr, eps):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        print(f"Number of params to train: {sum(param.numel() for param in params)}")
        opt = optim.AdamW(params, lr=lr, eps=eps)
        # self._lr_scheduler = get_constant_schedule_with_warmup(
        #     opt, num_warmup_steps=1000
        # )
        return opt

    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """
        learner_metrics["use_num_mini_batch"].append(rollouts.num_batches)
        learner_metrics["use_batch_size"].append(np.prod(batch["masks"].shape))

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()
        # Probably not necessary to have this as well. But just checking to be safe.
        self.optimizer.zero_grad()
        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch.get("rnn_build_seq_info", None),
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)
        batch_size, context_len, _ = action_loss.shape

        values = values[..., None]
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(values, batch["returns"], reduction="none")

        # Compute the loss mask to not include the samples from old rollout windows.
        # before_loss_mask = (
        #     torch.arange(start=0, end=context_len, device=device)
        #     .view(1, -1)
        #     .repeat(batch_size, 1)
        # )
        # before_counts = batch["observations"][FETCH_BEFORE_COUNTS_K].view(batch_size, 1)
        # before_loss_mask = (before_loss_mask >= before_counts).view(
        #     batch_size, context_len, 1
        # )

        # att_mask = batch["observations"][ATT_MASK_K]

        # loss_mask = att_mask * before_loss_mask
        # n_samples = loss_mask.sum()

        # def reduce_loss(loss):
        #     return (loss * loss_mask).sum() / n_samples
        reduce_loss = torch.mean
        action_loss, value_loss, dist_entropy = map(
            reduce_loss,
            (action_loss, value_loss, dist_entropy),
        )

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)

        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            record_min_mean_max(batch["returns"], "returns")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).to(self._dtype).mean()
                    + (ratio < (1.0 - self.clip_param)).to(self._dtype).mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(self.entropy_coef().detach())

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].to(self._dtype).mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (rollouts.current_policy_version - batch["policy_version"]).to(
                        self._dtype
                    ),
                    "policy_version_difference",
                )


class DistributedTransformerPPO(DecentralizedDistributedMixin, TransformerPPO):
    pass
