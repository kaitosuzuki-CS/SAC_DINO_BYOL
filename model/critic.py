import torch
import torch.nn as nn

from model.components import MLP


class SoftCritic(nn.Module):
    def __init__(self, encoder, action_dim, embed_dim, hps):
        super(SoftCritic, self).__init__()

        self._action_dim = action_dim
        self._embed_dim = embed_dim
        self._hps = hps

        self.encoder = encoder
        self.mlp = MLP(embed_dim + action_dim, hps.hidden_dim, 1)

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        B, C, H, W = x.shape

        x = self.encoder(x)

        x = torch.cat([x, a], dim=-1)
        x = self.mlp(x)

        return x
