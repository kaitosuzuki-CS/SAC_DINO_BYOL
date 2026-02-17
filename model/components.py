import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.act(out)
        out = self.out_layer(out)

        return out


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim, hps):
        super(ProjectionHead, self).__init__()

        self._embed_dim = embed_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hps.latent_dim if i != 0 else embed_dim, hps.latent_dim),
                    nn.ReLU(),
                )
                for i in range(hps.num_layers)
            ]
        )
        output_layer = nn.Linear(hps.latent_dim, hps.num_pseudo_labels)
        self.output_layer = nn.utils.parametrizations.weight_norm(
            output_layer, name="weight"
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = F.normalize(out, dim=-1)
        out = self.output_layer(out)

        return out


class Encoder(nn.Module):
    def __init__(self, obs_shape, embed_dim, hps):
        super(Encoder, self).__init__()

        self._obs_shape = obs_shape
        self._embed_dim = embed_dim
        self._hps = hps

        C, H, W = obs_shape
        self.in_conv = nn.Conv2d(C, hps.latent_dim, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hps.latent_dim,
                        hps.latent_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
                for _ in range(hps.num_layers - 1)
            ]
        )

        self.mlp = MLP(hps.latent_dim, hps.hidden_dim, embed_dim)
        self.output_layer = nn.Sequential(nn.LayerNorm(embed_dim), nn.Tanh())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.act(out)

        for layer in self.layers:
            out = layer(out)

        out = out.mean(dim=(-2, -1))
        out = self.mlp(out)
        out = self.output_layer(out)

        return out
