import torch
import torch.nn as nn
from torch.optim import Adam

from manipulation.algos.ppo.buffer import RolloutBuffer
from manipulation.algos.ppo.network import ActorCritic


class PPO:
    def __init__(self, obs_dim, action_dim, num_envs, cfg, device):
        self.cfg    = cfg
        self.device = device

        self.network = ActorCritic(
            obs_dim            = obs_dim,
            action_dim         = action_dim,
            actor_hidden_dims  = cfg.actor_hidden_dims,
            critic_hidden_dims = cfg.critic_hidden_dims,
            activation         = cfg.activation,
            init_noise_std     = cfg.init_noise_std,
        ).to(device)

        self.optimizer = Adam(self.network.parameters(), lr=cfg.learning_rate)

        self.buffer = RolloutBuffer(
            num_steps  = cfg.num_steps_per_env,
            num_envs   = num_envs,
            obs_dim    = obs_dim,
            action_dim = action_dim,
            device     = device
        )

        # ── Online obs normalizer (Welford) ───────────────────────────────────
        self.obs_mean  = torch.zeros(obs_dim, device=device)
        self.obs_var   = torch.ones(obs_dim, device=device)
        self.obs_count = torch.tensor(1.0, device=device)
    

    # ── Obs normalisation ─────────────────────────────────────────────────────
    def _update_obs_stats(self, obs):
        batch          = obs.reshape(-1, obs.shape[-1]).detach()
        n              = batch.shape[0]
        new_count      = self.obs_count + n
        new_mean       = (self.obs_count * self.obs_mean + batch.sum(0)) / new_count
        delta_old      = batch - self.obs_mean
        delta_new      = batch - new_mean
        new_var        = (self.obs_var * self.obs_count + (delta_old * delta_new).sum(0)) / new_count
        self.obs_mean  = new_mean
        self.obs_var   = new_var.clamp(min=1e-6)
        self.obs_count = new_count

    
    def normalize_obs(self, obs, update_stats=False):
        obs = obs.clamp(-100.0, 100.0)
        if update_stats:
            self._update_obs_stats(obs)
        return ((obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)
    

    # ── Rollout ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def collect_step(self, obs):
        action, log_prob, _, value = self.network.get_action_and_value(obs)
        return action, log_prob, value
    

    def insert(self, obs, actions, rewards, dones, values, log_probs):
        self.buffer.insert(obs, actions,rewards, dones, values, log_probs)

    
    @torch.no_grad()
    def compute_returns(self, last_obs):
        last_value = self.network.get_value(last_obs)
        self.buffer.compute_returns_and_advantages(
            last_value = last_value,
            gamma = self.cfg.gamma,
            lam = self.cfg.lam,
        )

    
    # ── Update ────────────────────────────────────────────────────────────────
    def update(self):
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.cfg.num_learning_epochs):
            for obs, actions, old_log_probs, advantages, returns, old_values in self.buffer.get_batches(self.cfg.num_mini_batches):
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(obs, action=actions)

                # ── Policy loss ───────────────────────────────────────────────
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                # ── Value loss (clipped) ──────────────────────────────────────
                values_clipped = old_values + torch.clamp(new_values - old_values, -self.cfg.clip_param, self.cfg.clip_param)
                loss_v_unclipped = (new_values - returns).pow(2)
                loss_v_clipped = (values_clipped - returns).pow(2)
                loss_value = 0.5 * torch.max(loss_v_unclipped, loss_v_clipped).mean()
 
                # ── Entropy loss ──────────────────────────────────────────────
                loss_entropy = -entropy.mean()

                loss = (
                    loss_policy +
                    0.5 * loss_value +
                    self.cfg.entropy_coef * loss_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()
                entropy_loss += loss_entropy.item()
                num_updates += 1

        self.buffer.reset()

        return {
            "loss/total":   total_loss   / num_updates,
            "loss/policy":  policy_loss  / num_updates,
            "loss/value":   value_loss   / num_updates,
            "loss/entropy": entropy_loss / num_updates,      
        }