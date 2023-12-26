import abc
from itertools import chain

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.utils.timing import g_timer

from habitat_transformers.policies.transformer_storage import (
    ATT_MASK_K,
    FETCH_BEFORE_COUNTS_K,
    HIDDEN_WINDOW_K,
    START_ATT_MASK_K,
    START_HIDDEN_WINDOW_K,
)
from habitat_transformers.policies.transformer_wrappers import (
    TransformerWrapper,
)
from habitat_transformers.policies.visual_encoders import Vc1Wrapper


class PolicyCore(nn.Module, abc.ABC):
    def __init__(self, obs_space, config):
        super().__init__()
        self._im_obs_space = spaces.Dict(
            {k: v for k, v in obs_space.items() if len(v.shape) == 3}
        )

        self._state_obs_space = spaces.Dict(
            {k: v for k, v in obs_space.items() if len(v.shape) == 1}
        )
        self._config = config
        self._hidden_size = self._config.hidden_dim
        self._is_blind = len(self._im_obs_space) == 0
        self._prefix_tokens_obs_k = config.prefix_tokens_obs_k

    @property
    @abc.abstractmethod
    def rnn_hidden_dim(self):
        pass

    @property
    def visual_encoder(self):
        return None

    @property
    def hidden_window_dim(self):
        return 512

    @abc.abstractmethod
    def forward(self, obs, rnn_hidden_states, masks, rnn_build_seq_info=None):
        pass

    @abc.abstractmethod
    def get_num_rnn_layers(self):
        pass

    @abc.abstractmethod
    def get_trainable_params(self):
        pass


class TransformerPolicyCore(PolicyCore):
    def __init__(self, obs_space, config):
        super().__init__(obs_space, config)
        self._train_decoder = config.train_decoder
        self._train_visual_encoder = config.train_visual_encoder
        self._use_vc1 = config.use_vc1

        self._debug_mode = self._config.debug_mode
        self.seq_model = TransformerWrapper(
            model_id=self._config.model_id,
            use_b16=config.use_b16,
            debug_mode=config.debug_mode,
        )

        self.context_len = self._config.context_len
        self._context_idxs = torch.arange(self.context_len)

        self._is_eval_mode = self._config.is_eval_mode
        self._context_window = None
        self._att_mask = None

        resnet_baseplanes = 32

        self.vis_encoder_net = Vc1Wrapper(
            self._im_obs_space, config.vc1_use_b16
        )

        if not self._train_visual_encoder:
            for param in self.vis_encoder_net.parameters():
                param.requires_grad = False

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.vis_encoder_net.output_shape),
                self.seq_model.d_model,
            ),
        )

        ac_hidden_size = config.ac_hidden_size
        self.action_proj = nn.Sequential(
            nn.Linear(self.seq_model.d_model, ac_hidden_size),
            nn.LayerNorm(ac_hidden_size),
            nn.ReLU(True),
            nn.Linear(ac_hidden_size, ac_hidden_size),
            nn.LayerNorm(ac_hidden_size),
            nn.ReLU(True),
        )

    @property
    def visual_encoder(self):
        return self.vis_encoder_net

    @property
    def hidden_window_dim(self):
        if self._debug_mode:
            return self.seq_model.d_model + 2
        else:
            return self.seq_model.d_model

    @property
    def rnn_hidden_dim(self):
        if self._is_eval_mode:
            return self.hidden_window_dim
        else:
            # No hidden dim.
            return 1

    def get_trainable_params(self):
        train_modules = [
            self.visual_fc.parameters(),
            self.action_proj.parameters(),
        ]
        if self.state_token_proj is not None:
            self.state_token_proj.parameters(),
        if self._train_visual_encoder:
            train_modules.append(self.vis_encoder_net.parameters())
        if self._train_decoder:
            train_modules.append(self.seq_model.parameters())

        return chain(*train_modules)

    def get_num_rnn_layers(self):
        return 1

    def _proj_vis_features(self, obs, visual_features):
        if self._linear_vis_proj:
            return visual_features
        else:
            state_features = [obs[k] for k in self._state_obs_space.keys()]

            if visual_features is None:
                hidden_window = torch.cat(state_features, dim=-1)
            elif len(state_features) == 0:
                hidden_window = visual_features
            else:
                hidden_window = torch.cat(
                    [visual_features, *state_features], dim=-1
                )

            return self.state_token_proj(hidden_window)

    @g_timer.avg_time("core.forward.eval.get_window", level=1)
    def _get_hidden_window_eval(self, obs, att_masks):
        lang = obs[self._prefix_tokens_obs_k]
        # Recompute all tokens.
        # Flatten to compute visual tokens.
        batch_shape = obs[list(obs.keys())[0]].shape[:2]

        if self._is_blind:
            visual_features = None
        elif PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in obs:
            # `visual_fc` expects an input of shape [batch_dim, hidden_dim] but
            # obs is of shape [batch_dim, context_size, hidden_dim]
            visual_features = self.visual_fc(
                obs[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY].flatten(
                    end_dim=1
                )
            ).view(*batch_shape, -1)
        else:
            # Flatten the batch dimension
            im_obs = {
                k: obs[k].flatten(end_dim=1) for k in self._im_obs_space.keys()
            }

            with g_timer.avg_time("core.visual_encode_eval", level=1):
                visual_features = self.vis_encoder_net(im_obs)
                visual_features = self.visual_fc(visual_features)
            # Re-expand to shape [batch_size, seq_len, obs_dim]
            visual_features = visual_features.view(*batch_shape, -1)

        hidden_window = self._proj_vis_features(obs, visual_features)

        # No need to cache any hidden state.
        latest_hidden_window = None

        # Embed the lang tokens. Observation is a window.
        self._context_idxs = self._context_idxs.to(att_masks.device)
        max_context_window = att_masks.shape[1]
        # We need to review the att_masks because during RL training it is
        # shape [batch_size, context_len, 1] and that extra 1 will produce the
        # wrong shape.
        last_valid_idxs = (
            self._context_idxs[:max_context_window]
            * att_masks.view(batch_shape)
        ).argmax(-1)
        lang_tokens = lang[torch.arange(batch_shape[0]), last_valid_idxs]

        fetch_before_counts_cpu = (
            obs[FETCH_BEFORE_COUNTS_K].cpu().numpy().tolist()
        )
        for batch_idx, fetch_before_count in enumerate(
            fetch_before_counts_cpu
        ):
            if fetch_before_count == 0:
                continue
            # Shift the tensor to the right to make room for the data from before. The data on the right is anyways bad
            hidden_window[batch_idx] = hidden_window[batch_idx].roll(
                shifts=fetch_before_count, dims=0
            )
            att_masks[batch_idx] = att_masks[batch_idx].roll(
                shifts=fetch_before_count, dims=0
            )
            hidden_window[batch_idx, :fetch_before_count] = obs[
                START_HIDDEN_WINDOW_K
            ][batch_idx, -fetch_before_count:]
            att_masks[batch_idx, :fetch_before_count] = obs[START_ATT_MASK_K][
                batch_idx, -fetch_before_count:
            ]

        return hidden_window, lang_tokens, latest_hidden_window, att_masks

    @g_timer.avg_time("core.forward.rollout.get_window", level=1)
    def _get_hidden_window_rollout(self, obs, att_masks):
        lang = obs[self._prefix_tokens_obs_k]
        # Compute from a hidden state. The hidden window should have the more
        # recent observations to the RIGHT (higher indices).
        hidden_window = obs[HIDDEN_WINDOW_K]

        if self._is_blind:
            latest_visual_feature = None
        elif (
            not self._train_visual_encoder
            and PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in obs
        ):
            latest_visual_feature = obs[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ]
        else:
            # Compute the most recent visual feature
            with g_timer.avg_time("core.visual_encode_rollout", level=1):
                latest_visual_feature = self.vis_encoder_net(obs)

        if latest_visual_feature is not None:
            latest_visual_feature = self.visual_fc(latest_visual_feature)

        latest_hidden_window = self._proj_vis_features(
            obs, latest_visual_feature
        )
        if self._debug_mode:
            debug_info = obs["debug_info"]
            latest_hidden_window = torch.cat(
                [latest_hidden_window, debug_info], -1
            )

        # Combine the latest visual token with the history of visual tokens.
        # (Add seq_dim to the `latest_hidden_window`)
        hidden_window = torch.cat(
            [hidden_window, latest_hidden_window.unsqueeze(1)], dim=1
        )

        if self._is_eval_mode:
            # Move the window forward.
            self._context_window = hidden_window[:, 1:]

        # Embed the lang tokens. Observation is only the current step.
        lang_tokens = lang
        hidden_window, att_masks, final_step_idxs = reverse_hidden_window(
            hidden_window, att_masks
        )

        return (
            hidden_window,
            lang_tokens,
            latest_hidden_window,
            att_masks,
            final_step_idxs,
        )

    def _prepare_eval_obs(self, obs, masks):
        # Not run during training.
        device = masks.device
        if self._context_window is None:
            n_envs = masks.shape[0]
            self._context_window = torch.zeros(
                # -1 because we will compute the most recent tokens.
                (n_envs, self.context_len - 1, self.hidden_window_dim),
                device=device,
            )
            self._att_mask = torch.zeros(
                (n_envs, self.context_len, 1), dtype=torch.bool, device=device
            )
        masks = masks.view(-1, 1, 1)
        assert len(masks.shape) == 3
        assert masks.shape[1] == 1 and masks.shape[2] == 1
        self._att_mask = self._att_mask.roll(shifts=-1, dims=1)
        self._context_window *= masks
        self._att_mask *= masks
        self._att_mask[:, -1] = True
        obs[ATT_MASK_K] = self._att_mask
        obs[HIDDEN_WINDOW_K] = self._context_window
        return obs

    def forward(self, obs, rnn_hidden_states, masks, rnn_build_seq_info=None):
        """
        :param masks: The episode `done` masks.
        :param rnn_build_seq_info: Ignored.
        :returns: Processed tensor of shape [batch_size, seq_len, hidden_size]
        """
        if self._is_eval_mode:
            obs = self._prepare_eval_obs(obs, masks)

        att_masks = obs[ATT_MASK_K]
        is_rollout = HIDDEN_WINDOW_K in obs

        final_step_idxs = None
        if is_rollout:
            # Rolling out the policy
            (
                hidden_out,
                lang_tokens,
                latest_hidden_window,
                att_masks,
                final_step_idxs,
            ) = self._get_hidden_window_rollout(obs, att_masks)
        else:
            # Evaluating the policy on a batch of trajectories.
            (
                hidden_out,
                lang_tokens,
                latest_hidden_window,
                att_masks,
            ) = self._get_hidden_window_eval(obs, att_masks)

        assert (
            lang_tokens.sum() != 0
        ), f"Bad lang tokens: {lang_tokens}, obs {obs['vocab_lang_goal']}"

        # Ensure all windows are right shape.
        assert (
            hidden_out.shape[1] == att_masks.shape[1]
        ), f"Got hidden window of shape {hidden_out.shape} and att masks of shape {att_masks.shape}"
        diagnose_prefix = "rollout" if is_rollout else "eval"

        with g_timer.avg_time(
            f"core.forward.{diagnose_prefix}.decoder", level=1
        ):
            hidden_out = self.seq_model.decode(
                lang_tokens, hidden_out, att_masks
            )

        with g_timer.avg_time(f"core.forward.{diagnose_prefix}.select_final"):
            if is_rollout:
                # We only want to return the last hidden state to predict the action
                sel_hidden_out = []
                check_att_masks = att_masks.cumprod(1)
                for i, final_step_idx in enumerate(final_step_idxs):
                    # Check the step is valid.
                    if (
                        self._debug_mode
                        and check_att_masks[i, final_step_idx] != 1
                    ):
                        raise ValueError(
                            f"Selecting bad action index {final_step_idx=} for {check_att_masks[i]}"
                        )
                    sel_hidden_out.append(hidden_out[i, final_step_idx])

                hidden_out = torch.stack(sel_hidden_out, dim=0)
                assert len(hidden_out.shape) == 2
            else:
                assert len(hidden_out.shape) == 3

        with g_timer.avg_time(
            f"core.forward.{diagnose_prefix}.action_proj", level=1
        ):
            hidden_out = self.action_proj(hidden_out)

        return hidden_out, latest_hidden_window


def reverse_hidden_window(hidden_window, att_masks):
    # Ensure all the True masks are sequential
    batch_size, context_len = att_masks.shape[:2]
    sub_att_masks = []
    sub_hidden_windows = []
    final_step_idxs = []
    for batch_idx in range(batch_size):
        batch_att_masks = att_masks[batch_idx]
        did_hit_true = False
        did_hit_false = False
        ep_start_idx = None

        for step_idx in range(context_len):
            if att_masks[batch_idx, step_idx].item():
                if not did_hit_true:
                    ep_start_idx = step_idx
                did_hit_true = True
            elif did_hit_true:
                # Once we start hitting True, we should only hit Trues.
                breakpoint()
                raise ValueError("Bad rollout att mask")
        assert ep_start_idx is not None

        roll_len = context_len - ep_start_idx
        final_step_idxs.append(roll_len - 1)
        # Roll the end to the start.
        sub_att_mask = att_masks[batch_idx].roll(roll_len, dims=0)
        sub_hidden_windows.append(
            hidden_window[batch_idx].roll(roll_len, dims=0)
        )

        sub_att_masks.append(sub_att_mask)

    att_masks = torch.stack(sub_att_masks, dim=0)
    hidden_window = torch.stack(sub_hidden_windows, dim=0)
    return hidden_window, att_masks, final_step_idxs
