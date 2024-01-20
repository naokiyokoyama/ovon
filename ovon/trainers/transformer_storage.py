import warnings
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.utils.common import get_num_actions, is_continuous_action_space
from torch.nn.utils.rnn import pad_sequence

ATT_MASK_K = "att_mask"
HIDDEN_WINDOW_K = "hidden_window"
START_HIDDEN_WINDOW_K = "start_hidden_window"
START_ATT_MASK_K = "start_att_mask"
FETCH_BEFORE_COUNTS_K = "fetch_before_counts"


def transpose_stack_pad_dicts(dicts_i):
    res = {}
    for k in dicts_i[0].keys():
        if isinstance(dicts_i[0][k], dict):
            res[k] = transpose_stack_pad_dicts([d[k] for d in dicts_i])
        else:
            res[k] = pad_sequence(
                [d[k] for d in dicts_i], batch_first=True, padding_value=0.0
            )

    return res


def np_dtype_to_torch_dtype(dtype):
    assert hasattr(torch, dtype.name)
    return getattr(torch, dtype.name)


def get_action_space_info(ac_space: spaces.Space) -> Tuple[Tuple[int], bool]:
    """
    :returns: The shape of the action space and if the action space is discrete.
      If the action space is discrete, the shape will be `(1,)`.
    """
    if is_continuous_action_space(ac_space):
        # Assume NONE of the actions are discrete
        return (
            (
                get_num_actions(
                    ac_space,
                ),
            ),
            False,
        )

    elif isinstance(ac_space, spaces.MultiDiscrete):
        return ac_space.shape, True
    elif isinstance(ac_space, spaces.Dict):
        num_actions = 0
        for _, ac_sub_space in ac_space.items():
            num_actions += get_action_space_info(ac_sub_space)[0][0]
        return (num_actions,), True

    else:
        # For discrete pointnav
        return (1,), True


class MinimalTransformerRolloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        num_recurrent_layers: int = 1,
        is_double_buffered: bool = False,
        action_shape: Optional[Tuple[int]] = None,
        discrete_actions: bool = False,
        device: bool = "cpu",
        dtype=torch.bfloat16,
        freeze_visual_feats=False,
    ):
        self._dtype = dtype
        self.context_length = actor_critic.context_len
        self.is_banded = actor_critic.banded_attention
        self.add_context_loss = actor_critic.add_context_loss

        self.is_first_update = True
        self.old_context_length = 0
        self._frozen_visual = (
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces
        )

        numsteps += self.context_length

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        self.vis_keys = {
            k
            for k in observation_space.spaces
            if len(observation_space.spaces[k].shape) == 3
        }

        if PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces:
            to_remove_sensor = self.vis_keys
        else:
            to_remove_sensor = []

        for sensor in observation_space.spaces:
            if sensor in to_remove_sensor:
                continue

            dtype = observation_space.spaces[sensor].dtype
            if "float" in dtype.name:
                dtype = np.dtype("float16")
            dtype = np_dtype_to_torch_dtype(dtype)

            shape = observation_space.spaces[sensor].shape

            self.buffers["observations"][sensor] = torch.zeros(
                (num_envs, numsteps + 1, *shape), dtype=dtype, device=device
            )

        self._recurrent_hidden_states_shape = (
            num_recurrent_layers,
            2,
            num_envs,
            actor_critic.num_heads,
            numsteps + 1,
            actor_critic.net.recurrent_hidden_size // actor_critic.num_heads,
        )
        self._recurrent_hidden_states = torch.zeros(
            self._recurrent_hidden_states_shape, device=device, dtype=self._dtype
        )

        self.buffers["rewards"] = torch.zeros(
            (num_envs, numsteps + 1, 1), device=device, dtype=self._dtype
        )
        self.buffers["value_preds"] = torch.zeros(
            (num_envs, numsteps + 1, 1), device=device, dtype=self._dtype
        )
        self.buffers["returns"] = torch.zeros(
            (num_envs, numsteps + 1, 1), device=device, dtype=self._dtype
        )

        self.buffers["action_log_probs"] = torch.zeros(
            (num_envs, numsteps + 1, 1), device=device, dtype=self._dtype
        )

        if action_shape is None:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            (num_envs, numsteps + 1, *action_shape), device=device, dtype=self._dtype
        )
        self.buffers["prev_actions"] = torch.zeros(
            (num_envs, numsteps + 1, *action_shape), device=device, dtype=self._dtype
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].to(self._dtype)
            self.buffers["prev_actions"] = self.buffers["prev_actions"].to(self._dtype)

        self.buffers["masks"] = torch.zeros(
            (num_envs, numsteps + 1, 1), dtype=torch.bool, device=device
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.num_steps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

        # The default device to torch is the CPU, so everything is on the CPU.
        self.device = torch.device(device)

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))
        if self._recurrent_hidden_states is not None:
            self._recurrent_hidden_states = self._recurrent_hidden_states.to(device)
        self.device = device

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        if self._frozen_visual and next_observations is not None:
            next_observations = {
                k: v for k, v in next_observations.items() if k not in self.vis_keys
            }

        next_step = dict(
            observations=next_observations,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index] + 1),
                next_step,
                strict=False,
            )

        if next_recurrent_hidden_states is not None:
            self._recurrent_hidden_states[
                :, :, env_slice, :, self.current_rollout_step_idxs[buffer_index] + 1
            ] = next_recurrent_hidden_states

        if len(current_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index]),
                current_step,
                strict=False,
            )

    def after_update(self):
        if self.context_length > 0:
            self.old_context_length = min(
                self.context_length, self.current_rollout_step_idx
            )
            self.buffers[:, 0 : self.old_context_length + 1] = deepcopy(
                self.buffers[
                    :,
                    self.current_rollout_step_idx
                    - self.old_context_length : self.current_rollout_step_idx
                    + 1,
                ]
            )
        else:
            self.old_context_length = 0
            self.buffers[:, 0] = self.buffers[:, self.current_rollout_step_idx]

        # self._recurrent_hidden_states = torch.zeros(
        #     *self._recurrent_hidden_states_shape,
        #     device=self.device, dtype=self._dtype
        # )

        self.current_rollout_step_idxs = [
            self.old_context_length for _ in self.current_rollout_step_idxs
        ]
        self.is_first_update = False

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][:, self.current_rollout_step_idx] = next_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][:, step]
                    + gamma
                    * self.buffers["value_preds"][:, step + 1]
                    * self.buffers["masks"][:, step + 1]
                    - self.buffers["value_preds"][:, step]
                )
                gae = delta + gamma * tau * gae * self.buffers["masks"][:, step + 1]
                self.buffers["returns"][:, step] = (  # type: ignore
                    gae + self.buffers["value_preds"][:, step]  # type: ignore
                )

        else:
            self.buffers["returns"][:, self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][:, step] = (
                    gamma
                    * self.buffers["returns"][:, step + 1]
                    * self.buffers["masks"][:, step + 1]
                    + self.buffers["rewards"][:, step]
                )

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def insert_first_observations(self, batch):
        if self._frozen_visual:
            batch = {k: v for k, v in batch.items() if k not in self.vis_keys}
        self.buffers["observations"][:, 0] = batch

    def get_current_step(self, env_slice, buffer_index):
        batch = self.buffers[
            env_slice,
            self.current_rollout_step_idxs[buffer_index],
        ]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idxs[buffer_index] - self.context_length,
            )
        else:
            start_idx = 0

        batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
            :,
            :,
            env_slice,
            :,
            start_idx + 1 : self.current_rollout_step_idxs[buffer_index] + 1,
        ]
        batch["masks"] = self.buffers[
            env_slice,
            start_idx : self.current_rollout_step_idxs[buffer_index] + 1,
        ]["masks"]

        return batch

    def get_last_step(self):
        batch = self.buffers[:, self.current_rollout_step_idx]

        if self.is_banded:
            start_idx = max(0, self.current_rollout_step_idx - self.context_length)
        else:
            start_idx = 0

        batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
            :,
            :,
            :,
            :,
            start_idx + 1 : self.current_rollout_step_idx + 1,
        ]
        batch["masks"] = self.buffers[
            :,
            start_idx : self.current_rollout_step_idx + 1,
        ]["masks"]

        # del self._recurrent_hidden_states
        # self._recurrent_hidden_states = None
        return batch

    def get_context_step(self, env_id=None, n_steps=None):
        n_steps = self.old_context_length if n_steps is None else n_steps
        if env_id is None:
            batch = self.buffers[:, :n_steps]
        else:
            batch = self.buffers[env_id : env_id + 1, :n_steps]
        dims = batch["masks"].shape[:2]
        batch.map_in_place(lambda v: v.flatten(0, 1))
        batch["rnn_build_seq_info"] = TensorDict({
            "dims": torch.from_numpy(np.asarray(dims)),
            "is_first": torch.tensor(True),
            "old_context_length": torch.tensor(n_steps),
        })
        return batch

    def update_context_data(
        self, value_preds, recurrent_hidden_states, env_id=None, n_steps=None
    ):
        n_steps = self.old_context_length if n_steps is None else n_steps
        env_id = slice(env_id, env_id + 1) if env_id is not None else slice()

        if value_preds is not None:
            self.buffers["value_preds"][env_id, :n_steps] = value_preds
        if recurrent_hidden_states is not None:
            self._recurrent_hidden_states[:, :, env_id, :, 1 : n_steps + 1] = (
                recurrent_hidden_states
            )

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(0)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_environments, num_mini_batch)
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        if advantages is not None:
            self.buffers["advantages"] = advantages

        # batches = self.buffers[
        #     np.random.permutation(num_environments), :self.current_rollout_step_idx
        # ]
        b_indexes = np.random.permutation(num_environments)
        batch_size = num_environments // num_mini_batch

        for inds in range(0, num_environments, batch_size):
            batch = self.buffers[
                b_indexes[inds : inds + batch_size], : self.current_rollout_step_idx
            ]
            batch["recurrent_hidden_states"] = torch.tensor([[]])

            if not self.add_context_loss and not self.is_first_update:
                keys = {
                    "action_log_probs",
                    "advantages",
                    "advantages",
                    "value_preds",
                    "value_preds",
                    "returns",
                    "is_coeffs",
                    "actions",
                }
                for k in keys & batch.keys():
                    batch[k] = batch[k][:, self.old_context_length :]

            batch.map_in_place(lambda v: v.flatten(0, 1))
            batch["rnn_build_seq_info"] = TensorDict({
                "dims": torch.from_numpy(
                    np.asarray([
                        min(inds + batch_size, num_environments) - inds,
                        self.current_rollout_step_idx,
                    ])
                ),
                "is_first": torch.tensor(self.is_first_update),
                "old_context_length": torch.tensor(self.old_context_length),
            })

            yield batch.to_tree()


class TransformerRolloutStorage(RolloutStorage):
    def __init__(
        self,
        *args,
        numsteps,
        num_envs,
        observation_space,
        actor_critic,
        **kwargs,
    ):
        self._frozen_visual = (
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces
        )
        if self._frozen_visual:
            # Remove the head RGB camera because we have the visual information
            # in the visual features
            observation_space = spaces.Dict(
                {k: v for k, v in observation_space.spaces.items() if k != "head_rgb"}
            )

        super().__init__(
            *args,
            numsteps=numsteps,
            num_envs=num_envs,
            observation_space=observation_space,
            actor_critic=actor_critic,
            **kwargs,
        )

        if hasattr(actor_critic, "_high_level_policy"):
            core = actor_critic._high_level_policy._policy_core
        else:
            # For the FlatPolicy
            core = actor_critic.policy_core

        self._context_len = core.context_len

        self.hidden_window = torch.zeros(
            2,
            2,
            num_envs,
            8,
            numsteps,
            64,
        )
        self.att_masks = torch.zeros(num_envs, numsteps, dtype=torch.bool)

        # The att masks BEFORE this rollout.
        self.before_start_att_masks = torch.zeros(
            num_envs, self._context_len, dtype=torch.bool
        )
        self._ep_boundry = torch.tensor([False, True])
        self._data_gen_count = 0
        self._max_batch_size = self._context_len * self._num_envs

    def to(self, device):
        super().to(device)
        self.hidden_window = self.hidden_window.to(device)
        self.att_masks = self.att_masks.to(device)
        self._ep_boundry = self._ep_boundry.to(device)

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if self._frozen_visual and next_observations is not None:
            next_observations = {
                k: v for k, v in next_observations.items() if k != "head_rgb"
            }
        super().insert(
            next_observations=next_observations,
            next_recurrent_hidden_states=None,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            next_masks=next_masks,
            buffer_index=buffer_index,
            **kwargs,
        )

        if next_recurrent_hidden_states is not None:
            write_step = slice(
                self.current_rollout_step_idxs[buffer_index]
                + 1
                - next_recurrent_hidden_states.shape[-2],
                self.current_rollout_step_idxs[buffer_index] + 1,
            )
            env_slice = slice(
                int(buffer_index * self._num_envs / self._nbuffers),
                int((buffer_index + 1) * self._num_envs / self._nbuffers),
            )

            self.hidden_window[:, :, env_slice, :, write_step] = (
                next_recurrent_hidden_states
            )

    def insert_first_observations(self, batch):
        if self._frozen_visual:
            batch = {k: v for k, v in batch.items() if k != "head_rgb"}
        super().insert_first_observations(batch)
        self.att_masks[:, 0] = False

    def after_update(self):
        """
        Copy over information from the end of the current rollout buffer into
        the rollout buffer start.
        """
        super().after_update()
        # self.hidden_window[:, : self._context_len] = (
        #     self.hidden_window[:, -self._context_len :].detach().clone()
        # )
        # self.before_start_att_masks.copy_(self.att_masks)

        # Clear the rest of the unused hidden window
        # self.hidden_window[:, self._context_len - 1 :] = 0.0
        # self.hidden_window[:, : self._context_len - 1] *= self.att_masks[
        #     :, :-1
        # ].unsqueeze(-1)

    @property
    def num_batches(self):
        return self._num_batches

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        self._data_gen_count += 1
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        # Naive way of doing things.
        self._num_batches = num_mini_batch
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            curr_slice = (slice(0, self.current_rollout_step_idx), inds)

            batch = self.buffers[curr_slice]
            if advantages is not None:
                batch["advantages"] = advantages[curr_slice]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]

            # batch.map_in_place(lambda v: v.flatten(0, 1))

            # batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
            #     device=self.device,
            #     build_fn_result=build_pack_info_from_dones(
            #         dones_cpu[
            #             0 : self.current_rollout_step_idx, inds.numpy()
            #         ].reshape(-1, len(inds)),
            #     ),
            # )

            yield batch.to_tree()

            # yield flatten_trajs(
            #     advantages,
            #     self.buffers,
            #     self._context_len,
            #     self.current_rollout_step_idx,
            #     self.hidden_window,
            #     self.before_start_att_masks,
            #     inds,
            # )

    def get_current_step(self, env_slice, buffer_index):
        buff_data = super().get_current_step(env_slice, buffer_index)

        cur_step = self.current_rollout_step_idxs[buffer_index]
        buff_data["observations"][HIDDEN_WINDOW_K] = self.hidden_window[
            :, :, env_slice, :, :cur_step
        ]
        buff_data["observations"][ATT_MASK_K] = self.att_masks[env_slice, :cur_step]
        return buff_data

    def get_last_step(self):
        buff_data = super().get_last_step()
        buff_data["observations"][HIDDEN_WINDOW_K] = self.hidden_window
        buff_data["observations"][ATT_MASK_K] = self.att_masks
        return buff_data


def flatten_trajs(
    advantages,
    buffers,
    context_len,
    current_rollout_step_idx,
    hidden_window,
    before_start_att_masks,
    inds,
    max_total_samples: Optional[int] = None,
) -> TensorDict:
    max_data_window_size = current_rollout_step_idx
    advantages.shape[0]
    device = advantages.device
    curr_slice = (
        slice(0, max_data_window_size),
        inds,
    )

    batch = buffers[curr_slice]
    batch["advantages"] = advantages[curr_slice]

    # Find where episode starts (when masks = False).
    inserted_start = {}
    ep_starts = torch.nonzero(~batch["masks"])[:, :2]
    ep_starts_cpu = ep_starts.cpu().numpy().tolist()
    for batch_idx in range(len(inds)):
        if [0, batch_idx] not in ep_starts_cpu:
            inserted_start[(0, batch_idx)] = True
            ep_starts_cpu.insert(0, [0, batch_idx])
    eps = []
    ep_starts_cpu = np.array(ep_starts_cpu)

    # Track the next start episode.
    batch_to_next_ep_starts = {}
    for batch_idx in range(len(inds)):
        batch_ep_starts = ep_starts_cpu[ep_starts_cpu[:, 1] == batch_idx][:, 0]
        batch_to_next_ep_starts[batch_idx] = {}
        for i in range(len(batch_ep_starts) - 1):
            batch_to_next_ep_starts[batch_idx][batch_ep_starts[i]] = batch_ep_starts[
                i + 1
            ]

    num_samples = 0
    fetch_before_counts = []
    for step_idx, batch_idx in ep_starts_cpu:
        step_idx_end = min(step_idx + context_len, max_data_window_size)

        next_ep_starts = batch_to_next_ep_starts[batch_idx]

        if step_idx in next_ep_starts:
            ep_intersect = step_idx_end - next_ep_starts[step_idx]
        else:
            ep_intersect = 0

        add_batch = batch.map(lambda x: x[step_idx:step_idx_end, batch_idx])

        att_mask = torch.ones(
            # The att mask needs to have same window size as the data batch.
            (step_idx_end - step_idx, 1),
            dtype=torch.bool,
            device=device,
        )
        assert att_mask.shape[0] == add_batch["masks"].shape[0], (
            f"Got att mask of shape {att_mask.shape} and masks of shape"
            f" {add_batch['masks'].shape}, trying to read from range"
            f" {step_idx}-{step_idx_end}, when step size is {max_data_window_size} cur"
            f" step idx {current_rollout_step_idx}, orig mask shape"
            f" {batch['masks'].shape}"
        )

        # Mask out steps that intersect with the next episode
        if ep_intersect > 0:
            att_mask[-ep_intersect:] = False

        # The trajectory should actually have some data.
        assert att_mask.sum() > 0, "Trajectory has no data"

        did_insert_start = inserted_start.get((step_idx, batch_idx), False)
        if did_insert_start:
            fetch_before_counts.append(ep_intersect)
        else:
            fetch_before_counts.append(0)

        # This is the max pad length
        num_samples += context_len

        add_batch["observations"][ATT_MASK_K] = att_mask
        add_batch["observations"][START_HIDDEN_WINDOW_K] = hidden_window[
            :, :, batch_idx, :, :context_len
        ]
        add_batch["observations"][START_ATT_MASK_K] = before_start_att_masks[
            batch_idx
        ].unsqueeze(-1)

        eps.append(add_batch)

        # Could the next seq addition put us over the max count threshold?
        if max_total_samples is not None and num_samples > (
            max_total_samples - context_len
        ):
            # Stop fetching trajectories.
            break

    assert len(eps) > 0, (
        "Collected no episodes from rollout, ensure episode horizon is shorter than"
        " rollout length."
    )
    ret_batch = transpose_stack_pad_dicts(eps)

    ret_batch["observations"][FETCH_BEFORE_COUNTS_K] = torch.tensor(
        fetch_before_counts, device=device
    )

    # Ensure every sample is accounted for in this data batch.
    num_samples = ret_batch["observations"][ATT_MASK_K].sum()
    len(inds) * max_data_window_size
    # This assert no longer works, because we might truncate the
    # episode collection based on the `max_total_samples`.
    # assert (
    #     expected_num_samples == num_samples
    # ), f"Incorrect number of data samples, got {num_samples},"
    # " expected {expected_num_samples}, Ep starts {ep_starts_cpu}"
    # " data window {max_data_window_size}"
    return TensorDict(ret_batch)
