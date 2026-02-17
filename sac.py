import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model.actor import Actor
from model.components import Encoder, ProjectionHead
from model.critic import SoftCritic
from utils.crop import Crop
from utils.replay_buffer import ReplayBuffer

parent_dir = Path(__file__).resolve().parent


class SAC_DINO:
    def __init__(self, env, hps, train_hps, device):
        self._env = env
        self._hps = hps
        self._train_hps = train_hps
        self._device = device

        self._init_hyperparameters()

        self.obs_shape, self.action_spec, self.action_shape = self._get_env_info()

        self.crop = Crop(
            self.num_global, self.global_crop_size, self.num_local, self.local_crop_size
        )

        self.replay_buffer = ReplayBuffer(
            self.buffer_capacity, self.obs_shape, self.action_shape, device
        )

        self.encoder = Encoder(self.obs_shape, hps.embed_dim, hps.encoder)
        self.projection = ProjectionHead(hps.embed_dim, hps.projection)

        self.target_encoder = Encoder(self.obs_shape, hps.embed_dim, hps.encoder)
        self.target_projection = ProjectionHead(hps.embed_dim, hps.projection)

        self.actor = Actor(self.encoder, self.action_shape[0], hps.embed_dim, hps.actor)

        self.critic1 = SoftCritic(
            self.encoder, self.action_shape[0], hps.embed_dim, hps.critic
        )
        self.critic2 = SoftCritic(
            self.encoder, self.action_shape[0], hps.embed_dim, hps.critic
        )

        self.target_critic1 = SoftCritic(
            self.encoder, self.action_shape[0], hps.embed_dim, hps.critic
        )
        self.target_critic2 = SoftCritic(
            self.encoder, self.action_shape[0], hps.embed_dim, hps.critic
        )

        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_shape).to(device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.center = torch.zeros(
            1, 1, hps.projection.num_pseudo_labels, requires_grad=False
        ).to(device)

        self._init_training_scheme()

    def _init_hyperparameters(self):
        self.loss_hps = self._train_hps.loss
        self.optimizer_hps = self._train_hps.optimizer
        self.data_hps = self._train_hps.data

        self.encoder_lr = float(self.optimizer_hps.encoder_lr)
        self.encoder_betas = tuple(map(float, self.optimizer_hps.encoder_betas))
        self.encoder_weight_decay = float(
            getattr(self.optimizer_hps, "encoder_weight_decay", 0)
        )

        self.actor_lr = float(self.optimizer_hps.actor_lr)
        self.actor_betas = tuple(map(float, self.optimizer_hps.actor_betas))
        self.actor_weight_decay = float(
            getattr(self.optimizer_hps, "actor_weight_decay", 0)
        )

        self.critic_lr = float(self.optimizer_hps.critic_lr)
        self.critic_betas = tuple(map(float, self.optimizer_hps.critic_betas))
        self.critic_weight_decay = float(
            getattr(self.optimizer_hps, "critic_weight_decay", 0)
        )

        self.alpha_lr = float(self.optimizer_hps.alpha_lr)
        self.alpha_betas = tuple(map(float, self.optimizer_hps.alpha_betas))
        self.alpha_weight_decay = float(
            getattr(self.optimizer_hps, "alpha_weight_decay", 0)
        )

        self.alpha = float(self._train_hps.alpha)
        self.gamma = float(self._train_hps.gamma)

        self.online_temp = float(self.loss_hps.online_temp)
        self.target_temp0 = float(self.loss_hps.target_temp0)
        self.target_temp1 = float(self.loss_hps.target_temp1)
        self.target_temp_warmup_timesteps = int(
            self.loss_hps.target_temp_warmup_timesteps
        )

        self.center_mom = float(self.optimizer_hps.center_mom)
        self.target_mom = float(self.optimizer_hps.target_mom)

        self.num_global = int(self.data_hps.num_global)
        self.global_crop_size = int(self.data_hps.global_crop_size)
        self.num_local = int(self.data_hps.num_local)
        self.local_crop_size = int(self.data_hps.local_crop_size)

        self.buffer_capacity = int(self.data_hps.buffer_capacity)

        self.total_timesteps = int(self._train_hps.total_timesteps)
        self.warmup_steps = int(self._train_hps.warmup_steps)
        self.start_training_steps = int(self._train_hps.start_training_steps)
        self.batch_size = int(self.data_hps.batch_size)

        self.center_update_freq = int(self.optimizer_hps.center_update_freq)
        self.target_update_freq = int(self.optimizer_hps.target_update_freq)

        self.checkpoints_dir = str(self._train_hps.checkpoints_dir)
        self.checkpoints_freq = int(self._train_hps.checkpoints_freq)

    def _init_training_scheme(self):
        self.encoder_optim = Adam(
            chain(self.encoder.parameters(), self.projection.parameters()),
            lr=self.encoder_lr,
            betas=self.encoder_betas,  # type: ignore
            weight_decay=self.encoder_weight_decay,
        )

        self.actor_optim = Adam(
            self.actor.mlp.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas,  # type: ignore
            weight_decay=self.actor_weight_decay,
        )

        self.critic1_optim = Adam(
            self.critic1.mlp.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas,  # type: ignore
            weight_decay=self.critic_weight_decay,
        )

        self.critic2_optim = Adam(
            self.critic2.mlp.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas,  # type: ignore
            weight_decay=self.critic_weight_decay,
        )

        self.alpha_optim = Adam(
            [self.log_alpha],
            lr=self.alpha_lr,
            betas=self.alpha_betas,  # type: ignore
            weight_decay=self.alpha_weight_decay,
        )

    def _init_weights(self):
        self.encoder.init_weights()
        self.projection.init_weights()
        self.actor.init_weights()
        self.critic1.init_weights()
        self.critic2.init_weights()

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projection.load_state_dict(self.projection.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self._freeze_parameters(self.target_encoder)
        self._freeze_parameters(self.target_projection)
        self._freeze_parameters(self.target_critic1)
        self._freeze_parameters(self.target_critic2)

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad = False

    def _soft_update(self, target_model, online_model, m):
        for target_param, param in zip(
            target_model.parameters(), online_model.parameters()
        ):
            target_param.data.copy_(m * target_param.data + (1.0 - m) * param.data)

    def _target_temp_schedule(self, t):
        if t < self.target_temp_warmup_timesteps:
            return (
                self.target_temp0
                + (self.target_temp1 - self.target_temp0)
                * t
                / self.target_temp_warmup_timesteps
            )

        return self.target_temp1

    def _target_momentum_schedule(self, t):
        return (
            1.0
            - (1.0 - self.target_mom)
            * (np.cos(np.pi * t / self.total_timesteps) + 1)
            / 2
        )

    def _get_env_info(self):
        obs_shape = self._hps.obs_shape
        action_spec = self._env.action_spec()
        action_shape = action_spec.shape

        return obs_shape, action_spec, action_shape

    def _move_to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.projection = self.projection.to(device)
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)
        self.target_encoder = self.target_encoder.to(device)
        self.target_projection = self.target_projection.to(device)
        self.target_critic1 = self.target_critic1.to(device)
        self.target_critic2 = self.target_critic2.to(device)

        print(f"Moved to: {device}")

    def q_forward(self, x, a):
        q1, q2 = self.target_critic1(x, a), self.target_critic2(x, a)
        return torch.min(q1, q2)

    def select_action(self, x):
        action, log_prob, _, _ = self.actor(x)
        return action, log_prob

    def dino_loss(self, global_views, local_views, num_updates):
        B, N_global, C, H_global, W_global = global_views.shape
        _, N_local, _, H_local, W_local = local_views.shape

        target_temp = self._target_temp_schedule(num_updates)

        global_views = global_views.view(B * N_global, C, H_global, W_global)
        local_views = local_views.view(B * N_local, C, H_local, W_local)

        global_logits = self.projection(self.encoder(global_views))
        global_logits = global_logits.view(B, N_global, -1)

        local_logits = self.projection(self.encoder(local_views))
        local_logits = local_logits.view(B, N_local, -1)

        logits = torch.cat([global_logits, local_logits], dim=1)
        with torch.no_grad():
            target_logits = self.target_projection(self.target_encoder(global_views))
            target_logits = target_logits.view(B, N_global, -1)

        log_probs = F.log_softmax(logits / self.online_temp, dim=-1)
        target_probs = F.softmax(
            (target_logits - self.center) / target_temp, dim=-1
        ).detach()

        loss = -(target_probs[:, :, None, :] * log_probs[:, None, :, :]).sum(dim=-1)

        mask = torch.ones(N_global, N_global + N_local, device=loss.device)
        mask.fill_diagonal_(0)

        loss = loss * mask[None]
        loss = loss.sum(dim=(1, 2)) / mask.sum()
        loss = loss.mean()

        return loss, logits, target_logits

    def update_parameters(self, num_updates):
        batch = self.replay_buffer.sample(self.batch_size)
        obs, action, reward, next_obs, done = batch

        global_views, local_views = self.crop(obs)
        obs_q = global_views[:, 0]
        next_obs = self.crop.single_crop(next_obs, self.crop.global_crop)

        mask = 1 - done.unsqueeze(1)
        reward = reward.unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.actor(next_obs)
            qf1_next_target, qf2_next_target = self.target_critic1(
                next_obs, next_state_action
            ), self.target_critic2(next_obs, next_state_action)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward + (mask * self.gamma * min_qf_next_target)

        qf1, qf2 = self.critic1(obs_q, action), self.critic2(obs_q, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        dino_loss, logits, target_logits = self.dino_loss(
            global_views, local_views, num_updates
        )

        self.encoder_optim.zero_grad()
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        qf_loss.backward()
        dino_loss.backward()
        self.encoder_optim.step()
        self.critic1_optim.step()
        self.critic2_optim.step()

        (
            pi,
            log_pi,
            _,
            _,
        ) = self.actor(obs_q)
        qf1_pi, qf2_pi = self.critic1(obs_q, pi), self.critic2(obs_q, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        with torch.no_grad():
            if num_updates % self.target_update_freq == 0:
                m = self._target_momentum_schedule(num_updates)
                self._soft_update(self.target_encoder, self.encoder, m)
                self._soft_update(self.target_projection, self.projection, m)
                self._soft_update(self.target_critic1, self.critic1, m)
                self._soft_update(self.target_critic2, self.critic2, m)

            if num_updates % self.center_update_freq == 0:
                self.center = self.center * self.center_mom + target_logits.mean(
                    dim=(0, 1), keepdim=True
                ) * (1 - self.center_mom)

    def train(self):
        self._init_weights()
        self._move_to_device(self._device)

        save_dir = os.path.join(parent_dir, self.checkpoints_dir)
        os.makedirs(save_dir, exist_ok=True)

        total_timesteps = 0
        num_episodes = 0
        num_updates = 0

        reward_log = []

        while total_timesteps < self.total_timesteps:
            num_episodes += 1

            episode_reward = 0
            episode_steps = 0
            done = False
            state, reward, done, _ = self._env.reset()

            while not done:
                if total_timesteps < self.warmup_steps:
                    action = np.random.uniform(
                        self.action_spec.minimum,
                        self.action_spec.maximum,
                        size=self.action_shape,
                    ).astype(self.action_spec.dtype)
                else:
                    with torch.no_grad():
                        _state = (
                            torch.as_tensor(state, device=self._device)
                            .unsqueeze(0)
                            .float()
                            / 255.0
                        )
                        _state = self.crop.single_crop(_state, self.crop.global_crop)
                        action, _ = self.select_action(_state)
                        action = (
                            action.detach().cpu().numpy().astype(self.action_spec.dtype)
                        )

                next_state, reward, done, _ = self._env.step(action)

                self.replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_timesteps += 1
                episode_steps += 1

                if self.replay_buffer.size > self.start_training_steps:
                    num_updates += 1
                    self.update_parameters(num_updates)

                    if num_updates % self.checkpoints_freq == 0:
                        torch.save(
                            {
                                "agent_state_dict": self.actor.state_dict(),
                                "total_timesteps": total_timesteps,
                                "num_episodes": num_episodes,
                                "num_updates": num_updates,
                            },
                            f"{save_dir}/checkpoints_{num_updates}.pt",
                        )

                if total_timesteps >= self.total_timesteps:
                    break

            reward_log.append(episode_reward)
            print(
                f"Episode: {num_episodes}, Reward: {episode_reward}, Steps: {episode_steps}, Total Steps: {total_timesteps}"
            )

        torch.save(
            {"agent_state_dict": self.actor.state_dict(), "reward_log": reward_log},
            f"{save_dir}/final_model.pt",
        )
