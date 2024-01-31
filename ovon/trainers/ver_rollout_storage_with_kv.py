import torch
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage


class VERRolloutStorageWithKVCache(VERRolloutStorage):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        head_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._aux_buffers["next_hidden_states"] = torch.zeros(
            self._num_envs,
            num_layers,
            2,  # key, value
            num_heads,
            max_context_length - 1,
            head_dim,
            device=self.buffers["recurrent_hidden_states"].device,
        )

        self._set_aux_buffers()
