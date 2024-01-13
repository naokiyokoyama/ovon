from typing import Tuple

import torch
import torch.nn as nn
import transformers
from habitat import logger

from ovon.models.llamarl.configuration_llamarl import LlamaRLConfig
from ovon.models.llamarl.modeling_llamarl import LlamaRLModel


class TransformerEncoder(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.model_name = config.model_name
        self.inter_episodes_attention = config.inter_episodes_attention
        self.reset_position_index = config.reset_position_index
        self.add_sequence_idx_embed = config.add_sequence_idx_embed
        self.n_layers = config.n_layers
        self.n_embed = config.n_hidden
        self.n_mlp_hidden = config.n_mlp_hidden
        self.n_head = config.n_heads
        self.activation = config.activation
        self.position_embed_type = config.position_embed_type
        self.depth_dropout_p = config.depth_dropout_p
        self.gated_residual = config.gated_residual

        self.context_len = config.context_len
        self.banded_attention = config.banded_attention
        self.orphan_steps_attention = config.orphan_steps_attention
        self.add_context_loss = config.add_context_loss
        self.max_position_embeddings = config.max_position_embeddings
        self.feats_proj = nn.Linear(input_size, self.n_embed)
        self.feats_out = nn.Linear(self.n_embed, self.n_embed)

        if self.model_name == "gpt":
            self.hf_config = transformers.GPT2Config(
                vocab_size=0,
                n_embd=self.n_embed,
                n_layer=self.n_layers,
                n_head=self.n_head,
            )

            self.model = transformers.GPT2Model(self.hf_config)
            self.model.wte.weight.requires_grad_(False)
        elif self.model_name == "llamarl":
            self.hf_config = LlamaRLConfig(
                hidden_size=self.n_embed,
                intermediate_size=self.n_mlp_hidden,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_head,
                hidden_act=self.activation,
                inter_episodes_attention=self.inter_episodes_attention,
                reset_position_index=self.reset_position_index,
                add_sequence_idx_embed=self.add_sequence_idx_embed,
                position_embed_type=self.position_embed_type,
                gated_residual=self.gated_residual,
                context_len=self.context_len,
                banded_attention=self.banded_attention,
                orphan_steps_attention=self.orphan_steps_attention,
                depth_dropout_p=self.depth_dropout_p,
                max_position_embeddings=self.max_position_embeddings,
            )

            self.model = LlamaRLModel(self.hf_config)
        else:
            raise ValueError(f"Unrecognized {self.model_name}")

        logger.info(f"Done loading {self.model_name} transformer.")
        self.cache = None

    def forward(
        self,
        feats,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info,
        full_rnn_state=False,
        **kwargs,
    ):
        # TODO: THIS IS EXTREMELY HACKY; self.actor_critic.net.state_encoder.num_steps
        # is set in dagger.py
        is_update_batch = masks.shape[1] == 1
        if is_update_batch:
            num_steps = self.num_steps

            new_shape = (feats.size(0) // num_steps, num_steps) + feats.size()[1:]
            feats = feats.view(new_shape)
            feats = self.feats_proj(feats)
            masks_seq = masks.view(new_shape[0], num_steps, -1).any(dim=-1).float()
            past_key_values = None
            position_ids = compute_position_ids(masks_seq)
            attention_mask = create_episodic_attn_mask(masks_seq)
        else:
            # Assuming that we are doing online inference
            feats = self.feats_proj(feats)
            assert feats.ndim == 2
            feats = feats.unsqueeze(1)  # (batch_size, 1, embed_dim)
            """
            masks_seq = masks.float()

            # Pop the oldest timestep from each sequence in the key-value cache
            # The cache has the following shape:
            # (batch_size, num_layers, 2, num_heads, seq_len, head_dim)
            rnn_hidden_states = rnn_hidden_states[:, :, :, :, 1:, :]

            # Extract the cache from the rnn_hidden_states
            rnn_hidden_states = rnn_hidden_states.to(feats.device)
            past_key_values = tensor_to_kv_cache(rnn_hidden_states)
            """
            if not masks[0, -1]:
                print("Resetting cache")
                self.cache = None

            past_key_values = self.cache

            position_ids = None
            attention_mask = None

        output = self.model(
            inputs_embeds=feats,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if is_update_batch:
            feats = output.last_hidden_state
            feats = feats.view(-1, feats.size(-1))
        else:
            # If we are doing online inference, we only need to return the hidden
            # states for the current time step
            # feats = feats.squeeze(1)

            # Extract the key-value cache from the output
            # rnn_hidden_states = kv_cache_to_tensor(output.past_key_values)

            feats = output.last_hidden_state[:, -1, :]
            self.cache = output.past_key_values

        return feats, rnn_hidden_states, []

    @property
    def num_recurrent_layers(self):
        return self.n_layers


def compute_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the position_ids tensor for the given mask.

    Args:
        mask (torch.Tensor): A mask tensor of shape (batch_size, seq_len) where a
            value of 0 indicates that the time step is the first time step in a sequence
            and a value of 1 indicates that the time step is not the first time step in
            a sequence.

    Returns:
        torch.Tensor: A position_ids tensor of shape (batch_size, seq_len).
    """
    # Calculate cumulative sum along the sequence dimension
    position_ids_cumulative = mask.cumsum(dim=1)

    # Reset the counter to 0 where the mask is 0 (start of a new sequence)
    # We use masked_fill_ for an in-place operation
    position_ids = position_ids_cumulative.masked_fill(mask == 0, 0)

    return position_ids


def kv_cache_to_tensor(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
) -> torch.Tensor:
    """
    Convert a tuple of key-value cache placeholders to a tensor.

    Args:
        kv_cache (Tuple[Tuple[torch.Tensor, torch.Tensor], ...]): A tuple of tuples
            containing key-value cache placeholders.

    Returns:
        torch.Tensor: A tensor representation of the key-value cache.
    """
    # Infer dimensions from the input kv_cache
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape

    # Initializing a tensor to store the converted kv_cache
    kv_tensor = torch.zeros(batch_size, num_layers, 2, num_heads, seq_len, head_dim)

    for layer_idx, (k, v) in enumerate(kv_cache):
        kv_tensor[:, layer_idx, 0, :, :, :] = k
        kv_tensor[:, layer_idx, 1, :, :, :] = v

    return kv_tensor


def tensor_to_kv_cache(
    kv_tensor: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Convert a tensor representation of key-value cache back to its original tuple
    format.

    Args:
        kv_tensor (torch.Tensor): A tensor representation of the key-value cache.

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...]: The original tuple format of
            key-value cache placeholders.
    """
    # Infer dimensions from the input kv_tensor
    batch_size, num_layers, _, num_heads, seq_len, head_dim = kv_tensor.shape

    kv_cache = tuple(
        (kv_tensor[:, layer_idx, 0, :, :, :], kv_tensor[:, layer_idx, 1, :, :, :])
        for layer_idx in range(num_layers)
    )
    return kv_cache


def create_episodic_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Create an episodic attention mask for the given mask.

    Args:
        mask (torch.Tensor): A mask tensor of shape (batch_size, seq_len) where a
            value of 0 indicates that the time step is the first time step in a sequence
            and a value of 1 indicates that the time step is not the first time step in
            a sequence.
    Returns:
        torch.Tensor: An episodic attention mask of shape (batch_size, seq_len,
            seq_len). This tensor is filled with values of either 0.0 or '-inf'. A value
            of 0.0 at a position (i, j, k) indicates that, within the same episode, the
            model is allowed to attend from time step j to time step k in batch i. A
            value of '-inf' blocks attention at those positions. The mask dynamically
            segments the input sequences into episodes based on the input mask,
            allowing attention only within these episodes and not across their
            boundaries.

    """
    batch_size, seq_len = mask.shape
    attn_mask = torch.full((batch_size, seq_len, seq_len), float("-inf"))

    for i in range(batch_size):
        episode_start = 0
        for j in range(seq_len):
            if mask[i, j] == 0:
                episode_start = j  # Start of a new episode
            attn_mask[i, j, episode_start : j + 1] = (
                0.0  # Allow attention within the episode
            )

    return attn_mask
