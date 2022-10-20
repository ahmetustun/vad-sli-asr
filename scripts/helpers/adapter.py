import torch.nn as nn
from transformers.activations import ACT2FN


class Adapter(nn.Module):
    """Conventional Transformer adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.adapter_dim = config.bottleneck_adapter_dim
        self.act_fn = ACT2FN[config.bottleneck_adapter_act]

        self.down = nn.Linear(self.input_dim, self.adapter_dim)
        self.up = nn.Linear(self.adapter_dim, self.input_dim)
        self.layer_norm = nn.LayerNorm(self.input_dim, eps=self.config.layer_norm_eps)

        self.init_linear_layer(self.down)
        self.init_linear_layer(self.up)

    def forward(self, x):
        z = self.down(x)
        z = self.act_fn(z)
        z = self.up(z)
        z = x + self.layer_norm(z)
        return z

    def init_linear_layer(self, linear_layer):
        """Initializes the given linear module as explained in adapter paper."""
        almost_zero = 1e-5
        delta = 1e-6
        nn.init.uniform_(linear_layer.weight, almost_zero - delta, almost_zero + delta)
        nn.init.uniform_(linear_layer.bias, almost_zero - delta, almost_zero + delta)


class CnnAdapter(nn.Module):

    def __init__(self, config):
        super(CnnAdapter, self).__init__()
        self.config = config
        self.conv = nn.Conv1d(
            self.config.cnn_adapter_in_conv_dim,
            self.config.cnn_adapter_out_conv_dim,
            kernel_size=config.cnn_adapter_kernel,
            stride=config.cnn_adapter_stride,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(self.config.cnn_adapter_in_conv_dim, elementwise_affine=True)
        self.do_norm = self.config.cnn_adapter_do_norm

    def forward(self, x):
        if self.do_norm:
            x = x.transpose(-2, -1)
            x = self.layer_norm(x)
            x = x.transpose(-2, -1)
        y = self.conv(x)
        y += x
        return y