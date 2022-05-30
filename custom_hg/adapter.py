import torch.nn as nn
from transformers.activations import ACT2FN


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.adapter_dim = config.adapter_dim
        self.act_fn = ACT2FN[config.adapter_act]

        self.down = nn.Linear(self.input_dim, self.adapter_dim, std=self.initializer_range)
        self.up = nn.Linear(self.adapter_dim, self.input_dim, std=self.initializer_range)
        self.layernorm = nn.LayerNorm(config.input_dim)

    def forward(self, x):
        z = self.down(x)
        z = self.act_fn(z)
        z = self.up(z)
        return self.layernorm(z)