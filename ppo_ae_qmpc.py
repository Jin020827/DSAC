import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def _to_action_dim(action_shape):
    if isinstance(action_shape, (list, tuple)):
        return int(np.prod(action_shape))
    return int(action_shape)


def _squashed_log_prob(dist, raw_action, action):
    eps = 1e-6
    action = action.clamp(eps, 1.0 - eps)
    log_prob = dist.log_prob(raw_action) - torch.log(action) - torch.log(1.0 - action)
    return log_prob.sum(-1, keepdim=True)


class RolloutBuffer_qMpc:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.q_weights = []
        self.velocities = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    @property
    def count(self):
        return len(self.rewards)

    def add(self, state, q_weight, velocity, action, log_prob, value, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.q_weights.append(np.asarray(q_weight, dtype=np.float32).reshape(1,))
        self.velocities.append(np.asarray(velocity, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))


class ActorCritic_qMpc(nn.Module):
    def __init__(self, action_shape, hidden_dim):
        super().__init__()
        self.action_dim = _to_action_dim(action_shape)
        input_dim = 30 + 1 + 3
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.apply(weight_init)

    def _features(self, obstacle, q_weight, velocity):
        return torch.cat((obstacle, q_weight, velocity), dim=-1)

    def forward(self, obstacle, q_weight, velocity):
        feature = self._features(obstacle, q_weight, velocity)
        mean = self.actor(feature)
        value = self.critic(feature)
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 2.0)
        return mean, log_std, value

    def act(self, obstacle, q_weight, velocity, deterministic=False):
        mean, log_std, value = self.forward(obstacle, q_weight, velocity)
        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + torch.randn_like(mean) * log_std.exp()
        action = torch.sigmoid(raw_action)
        dist = Normal(mean, log_std.exp())
        log_prob = _squashed_log_prob(dist, raw_action, action)
        return action, log_prob, value

    def evaluate_actions(self, obstacle, q_weight, velocity, action):
        eps = 1e-6
        action = action.clamp(eps, 1.0 - eps)
        raw_action = torch.log(action) - torch.log(1.0 - action)
        mean, log_std, value = self.forward(obstacle, q_weight, velocity)
        dist = Normal(mean, log_std.exp())
        log_prob = _squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, value


class PPO_Ae_qMpc(object):
    """PPO actor-critic algorithm for Q matrix adaptation."""

    def __init__(
        self,
        env,
        num_env,
        obs_shape,
        action_shape,
        writer=None,
        batch_size=64,
        hidden_dim=512,
        discount=0.99,
        lr=3e-4,
        seed=0,
        mode="train",
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=1024,
        **kwargs,
    ):
        self.active = env.index == 0
        self.mode = mode
        self.writer = writer
        self.epoch = 0
        self.num_env = num_env
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.total_it = 0
        self.action_bound = [[0.0, 1.0]]
        self._pending_action_info = None

        if self.active:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            self.actor_critic = ActorCritic_qMpc(action_shape, hidden_dim).to(device)
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
            self.rollout = RolloutBuffer_qMpc()
        else:
            self.actor_critic = None
            self.optimizer = None
            self.rollout = None

    def train(self, training=True):
        if self.active:
            self.actor_critic.train(training)

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return allocated, cached
        return 0, 0

    def generate_q_action(self, env, state_list, velocity_rms):
        if not self.active:
            return None

        s_list, q_list, velocity_list = [], [], []
        for state in state_list:
            s_list.append(state[0])
            q_list.append(state[1])
            velocity_list.append(state[2])

        s_array = np.asarray(s_list, dtype=np.float32)
        q_array = np.asarray(q_list, dtype=np.float32)
        velocity_array = velocity_rms.normalize(np.asarray(velocity_list, dtype=np.float32))

        state_tensor = torch.from_numpy(s_array).float().to(device)
        q_tensor = torch.from_numpy(q_array).float().to(device)
        velocity_tensor = torch.from_numpy(velocity_array).float().to(device)

        deterministic = self.mode == "test"
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.act(
                state_tensor, q_tensor, velocity_tensor, deterministic=deterministic
            )

        action_array = action.cpu().numpy()
        action_array = np.clip(action_array, 0.0, 1.0)
        if self.mode == "train":
            self._pending_action_info = {
                "log_prob": log_prob.cpu().numpy().reshape(-1),
                "value": value.cpu().numpy().reshape(-1),
            }
        return action_array

    def step(self, exp_list):
        if not self.active or self.mode != "train":
            return
        if self._pending_action_info is None:
            return

        log_probs = self._pending_action_info["log_prob"]
        values = self._pending_action_info["value"]
        for idx, exp in enumerate(exp_list):
            if exp is None:
                continue
            O_z, O_q, O_velocity, action, _, _, _, reward, not_done = exp
            done = 1.0 - float(not_done)
            safe_idx = min(idx, len(log_probs) - 1)
            self.rollout.add(
                O_z,
                O_q,
                O_velocity,
                action,
                log_probs[safe_idx],
                values[safe_idx],
                reward,
                done,
            )
        self._pending_action_info = None

    def _prepare_batch(self, velocity_rms):
        states = torch.FloatTensor(np.asarray(self.rollout.states)).to(device)
        q_weights = torch.FloatTensor(np.asarray(self.rollout.q_weights)).to(device)
        velocities = velocity_rms.normalize(np.asarray(self.rollout.velocities, dtype=np.float32))
        velocities = torch.FloatTensor(velocities).to(device)
        actions = torch.FloatTensor(np.asarray(self.rollout.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.asarray(self.rollout.log_probs)).unsqueeze(1).to(device)
        values_np = np.asarray(self.rollout.values, dtype=np.float32)
        rewards_np = np.asarray(self.rollout.rewards, dtype=np.float32)
        dones_np = np.asarray(self.rollout.dones, dtype=np.float32)

        returns = np.zeros_like(rewards_np, dtype=np.float32)
        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards_np))):
            mask = 1.0 - dones_np[t]
            delta = rewards_np[t] + self.discount * next_value * mask - values_np[t]
            gae = delta + self.discount * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values_np[t]
            next_value = values_np[t]

        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return states, q_weights, velocities, actions, old_log_probs, returns, advantages

    def samplefromrollout(self, velocity_rms):
        return self._prepare_batch(velocity_rms)

    def update_actor_critic(self, states, q_weights, velocities, actions,
                            old_log_probs, returns, advantages, mb_idx):
        new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
            states[mb_idx], q_weights[mb_idx], velocities[mb_idx], actions[mb_idx]
        )

        ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
        unclipped = ratio * advantages[mb_idx]
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio
        ) * advantages[mb_idx]

        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns[mb_idx])
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.total_it += 1

        return policy_loss, value_loss, entropy_loss

    def update(self, rollout_batch):
        states, q_weights, velocities, actions, old_log_probs, returns, advantages = rollout_batch
        batch_size = states.size(0)
        last_losses = None

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, self.mini_batch_size):
                mb_idx = indices[start:start + self.mini_batch_size]
                last_losses = self.update_actor_critic(
                    states, q_weights, velocities, actions,
                    old_log_probs, returns, advantages, mb_idx
                )

        return last_losses

    def myupdate(self, velocity_rms):
        rollout_batch = self.samplefromrollout(velocity_rms)
        last_losses = self.update(rollout_batch)
        self.rollout.clear()
        return last_losses

    def learn(self, velocity_rms):
        if not self.active or self.rollout.count < self.mini_batch_size:
            return

        print(f"[PPO-Q] training rollout={self.rollout.count}, epoch={self.epoch}")
        last_losses = self.myupdate(velocity_rms)

        if self.writer is not None and last_losses is not None:
            policy_loss, value_loss, entropy_loss = last_losses
            self.writer.add_scalar("loss/q_policy_loss", policy_loss.item(), self.epoch)
            self.writer.add_scalar("loss/q_value_loss", value_loss.item(), self.epoch)
            self.writer.add_scalar("loss/q_entropy", entropy_loss.item(), self.epoch)

    def save(self, epoch, policy_path):
        if not self.active:
            return
        os.makedirs(policy_path, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, policy_path + "/actor_critic_{:03d}".format(epoch))

    def load(self, model_file, mode, policy_epoch_index=0):
        if not self.active:
            return 0
        checkpoint_file = model_file + "/actor_critic_" + "{:03}".format(policy_epoch_index)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        self.epoch = checkpoint["epoch"]
        self.mode = mode
        if mode == "test":
            self.actor_critic.eval()
        return starting_epoch
