import os

import torch
from torch import nn

NUM_HEADS = int(os.environ.get("OVON_XATTN_NUM_HEADS", 12))


class CrossAttention(nn.Module):
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        num_heads: int = NUM_HEADS,
        dropout: float = 0.1,
    ) -> None:
        """
        Meant for fusion of two different modalities, x1 being language embeddings and
        x2 being visual embeddings.

        Args:
            x1_dim: Dimension of the first input (language)
            x2_dim: Dimension of the second input (visual)
            embed_dim: Dimension of the embedding space
            num_heads: Number of heads for the multihead attention
            dropout: Dropout rate for the multihead attention
        """
        super(CrossAttention, self).__init__()

        embed_dim = num_heads * 64

        # Linear layers to project x1 and x2 into embedding space
        self.proj1 = nn.Linear(x1_dim, embed_dim)
        self.proj2 = nn.Linear(x2_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_size = embed_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch_size, x1_dim] tensor (language)
            x2: [batch_size, x2_dim] tensor (visual)

        Returns:
            [batch_size, embed_dim] tensor
        """

        # Project x1 and x2 into the embedding space
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        # Reshape the tensors to be [seq_len, batch_size, embed_dim], where seq_len is 1
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)

        # Perform the cross-attention calculation
        # output: [1, batch_size, embed_dim]
        output, _ = self.multihead_attn(query=x1, key=x2, value=x2)

        # Apply Layer Normalization
        # output: [1, batch_size, embed_dim]
        output = self.norm(output)

        # Squeeze the sequence length dimension
        # output: [batch_size, embed_dim]
        output = output.squeeze(0)

        return output
