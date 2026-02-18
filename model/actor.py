import torch
import torch.nn as nn
from torch.distributions import Normal

from model.components import MLP


class Actor(nn.Module):
    def __init__(self, encoder, action_dim, embed_dim, hps):
        super(Actor, self).__init__()

        self._action_dim = action_dim
        self._embed_dim = embed_dim
        self._hps = hps

        self.encoder = encoder
        self.mlp = MLP(embed_dim, hps.hidden_dim, 2 * action_dim)

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.encoder(x).detach()
        out = self.mlp(out)

        mu, logvar = out.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)

        dist = Normal(mu, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mu, logvar
