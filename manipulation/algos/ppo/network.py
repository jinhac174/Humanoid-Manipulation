import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.15,
    ):
        super().__init__()

        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(obs_dim,              actor_hidden_dims[0]), act_fn,
            nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]), act_fn,
            nn.Linear(actor_hidden_dims[1], actor_hidden_dims[2]), act_fn,
            nn.Linear(actor_hidden_dims[2], action_dim),
        )

        # ── Critic ────────────────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(obs_dim,               critic_hidden_dims[0]), act_fn,
            nn.Linear(critic_hidden_dims[0], critic_hidden_dims[1]), act_fn,
            nn.Linear(critic_hidden_dims[1], critic_hidden_dims[2]), act_fn,
            nn.Linear(critic_hidden_dims[2], 1),
        )

        # learned log std — not obs-dependent, standard for manipulation
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(init_noise_std))
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)

        # small gain on actor output — policy starts near zero actions
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # standard gain on critic output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        mean  = torch.tanh(self.actor(obs))
        std   = self.log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value