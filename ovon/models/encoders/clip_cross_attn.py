import torch
import torch.nn.functional as F
from torch import Tensor, nn


def forward_attn_avg_pool(self, x):
    """Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138"""

    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    attnpool = self.attnpool(x)
    avgpool = self.adaptive_avgpool(x)
    out = torch.cat([attnpool, avgpool], dim=1)

    return out


class CLIPCrossAttentionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_encoding_size = 1024
        self.rgb_linear = nn.Linear(self.clip_encoding_size, self.clip_encoding_size)
        self.rgb_kv = nn.Linear(self.clip_encoding_size, self.clip_encoding_size * 2)

        self.text_linear = nn.Linear(self.clip_encoding_size, self.clip_encoding_size)
        self.register_buffer(
            "_scale", torch.tensor(1 / (self.clip_encoding_size ** 0.5))
        )

    def _attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        :param q: a query tensor with shape (batch_size, embed_dim)
        :param k: a key tensor with shape (batch_size, embed_dim)
        :param v: a value tensor with shape (batch_size, embed_dim)
        :return: cross attention output with shape (batch_size, embed_dim)
        """
        logits = (q * k).sum(1)
        attn = F.softmax(logits * self._scale, dim=0)
        out = attn.unsqueeze(-1) * v

        return out

    def forward(self, clip_rgb, clip_text):
        text_q = self.text_linear(clip_text)
        embedded_rgb = self.rgb_linear(clip_rgb)
        rgb_k, rgb_v = torch.split(
            self.rgb_kv(embedded_rgb), self.clip_encoding_size, dim=1
        )
        out = self._attn(text_q, rgb_k, rgb_v)

        return out
