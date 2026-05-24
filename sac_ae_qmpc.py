import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
from encoder import make_encoder
from decoder import make_decoder
# from torch.autograd import Variable  # 已弃用，不再需要
from Logger import Logger
import logging
from datetime import datetime
import itertools
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    # mu_1 = mu[:, 0].unsqueeze(-1)
    mu_2 = mu[:]
    mu_2 = torch.sigmoid(mu_2)
    mu = mu_2
    if pi is not None:
        pi_2 = pi[:]
        pi_2 = torch.sigmoid(pi_2)
        pi = pi_2
    if log_pi is not None:
        pi_clamped = pi.clamp(1e-6, 1.0 - 1e-6)
        log_pi -= (torch.log(pi_clamped) + torch.log(1.0 - pi_clamped)).sum(-1, keepdim=True)
    return mu, pi, log_pi

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    """MLP actor network for Q matrix adaptation."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
        feature_dim=32
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # 输入：编码器特征 + Q权重(1) + 速度(3)，输出：1个Q权重值
        input_dim = 30 + 1 + 3  # 编码器特征 + Q权重(1) + 速度(3)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape * 2)  # 输出1个Q权重值的均值和标准差
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, q_weights, velocity, compute_pi=True, compute_log_pi=True, detach=False):
        a = torch.cat((x, q_weights, velocity), dim=-1)
        mu, log_std = self.trunk(a).chunk(2, dim=-1) # 神经网络计算
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, hidden_dim, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_shape, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feature, action):
        assert feature.size(0) == action.size(0)
        obs_action = torch.cat([feature, action], dim=1)

        return self.trunk(obs_action)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, hidden_dim, action_shape, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()

        self.q1 = QFunction(30 + 1 + 3, hidden_dim, action_shape)  # 编码器特征 + Q权重(1) + 速度(3)
        self.q2 = QFunction(30 + 1 + 3, hidden_dim, action_shape)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, q_weights, velocity, action, detach=False):
        v = torch.cat((x, q_weights), dim=-1)
        v = torch.cat((v, velocity), dim=-1)

        q1 = self.q1(v, action)
        q2 = self.q2(v, action)

        return q1, q2

class SAC_Ae_qMpc(object):
    """SAC+AE algorithm for Q parameter adjustment. 输入：图像+Q值+速度，输出：1维Q值"""
    def __init__(
        self,
        env,
        num_env, 
        obs_shape,
        action_shape,
        writer = None,
        batch_size=256,
        replayer_buffer=1e4,
        init_steps=100, 
        hidden_dim=512,
        discount=0.99,
        init_temperature=0.1,
        alpha_beta=0.5,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_beta=0.9,
        critic_target_update_freq=1,
        lr=1e-3,
        tau=0.005,
        num_layers=4,
        num_filters=32,
        lam_a=-1.,
        lam_s=-1., 
        eps_s=1.,
        seed=0,
        mode='train'
    ):
        if env.index == 0:
            self.f_rec_loss = './log/' + '/rec_loss.log'
            self.L = Logger(self.f_rec_loss, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
            self.batch_size = batch_size
            self.action_shape = 1  # 输出避障权重
            self.actor_update_freq = actor_update_freq
            self.critic_target_update_freq = critic_target_update_freq
            self.discount = discount
            self.tau = tau
            self.lam_a = lam_a
            self.lam_s = lam_s
            self.eps_s = eps_s
            self.init_steps = init_steps
            self.update_flag = False
            self.mode = mode
            self.num_env = num_env
            self.writer = writer
            self.epoch = 0
            self.epoch_count = 0
            self.learning_time = 200
            np.random.seed(seed) 
            torch.cuda.manual_seed(seed)  
            torch.backends.cudnn.deterministic = True
            self.replayer_buffer = replayer_buffer
            self.replayer = utils.ReplayBuffer_qmpc(self.replayer_buffer)
            self.action_bound = [[0 , 1.0]]
            self.total_it = 0
            self.actor = Actor(
                obs_shape, action_shape, hidden_dim, actor_log_std_min, actor_log_std_max, num_layers, num_filters
            ).to(device)
            self.critic = Critic(
                obs_shape, hidden_dim, action_shape, num_layers, num_filters
            ).to(device)
            self.critic_target = Critic(
                obs_shape, hidden_dim, action_shape, num_layers, num_filters
            ).to(device)

            self.critic_target.load_state_dict(self.critic.state_dict())
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
            self.log_alpha.requires_grad = True
            self.target_entropy = -np.prod(action_shape)

            # optimizers
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr, betas=(actor_beta, 0.999)
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr, betas=(critic_beta, 0.999)
            )
            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=lr/10, betas=(alpha_beta, 0.999)
            )

            self.train()
            self.critic_target.train()
        else:
            pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def clear_gpu_memory(self):
        """清理GPU显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_gpu_memory_usage(self):
        """获取GPU显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return allocated, cached
        return 0, 0

    def generate_q_action(self, env, state_list, velocity_rms):
        if env.index == 0:
            s_list, q_list, velocity_list = [], [], []
            for i in state_list:
                s_list.append(i[0])
                q_list.append(i[1])
                velocity_list.append(i[2])  
            
            s_list = np.asarray(s_list) # 无人机障碍物位置向量
            q_list = np.asarray(q_list) # 代价函数权重
            velocity_list = np.asarray(velocity_list) # 无人机速度
            velocity_list = velocity_rms.normalize(velocity_list) # 速度归一化
            
            state_tensor = torch.from_numpy(s_list).float().to(device)
            q_tensor = torch.from_numpy(q_list).float().to(device)
            velocity_tensor = torch.from_numpy(velocity_list).float().to(device)
            
            action_bound = np.array(self.action_bound)
            if self.mode == 'train':
                mu, pi, _, _ = self.actor(state_tensor, q_tensor, velocity_tensor, compute_log_pi=False)
                pi = pi.cpu().data.numpy()
                scaled_action = copy.deepcopy(pi)
                for j in range(1):
                    scaled_action[:, j] = np.clip(scaled_action[:, j], a_min=action_bound[j][0], a_max=action_bound[j][1])
            elif self.mode == 'test':
                mu, _, _, _ = self.actor(state_tensor, q_tensor, velocity_tensor, compute_log_pi=False)
                mu = mu.cpu().data.numpy()
                scaled_action = copy.deepcopy(mu)
                for j in range(1):
                    scaled_action[:, j] = np.clip(scaled_action[:, j], a_min=action_bound[j][0], a_max=action_bound[j][1])
        else:
            scaled_action = None

        return scaled_action

    def update_critic(self, state_tensor, q_tensor, velocity_tensor, action, reward, n_state_tensor, n_q_tensor, n_velocity_tensor, not_done):
        """更新Critic网络，使用图像+Q值+速度输入"""
        # 复制下一个状态张量
        n_state_tensor_copy = n_state_tensor.clone()
        n_q_tensor_copy = n_q_tensor.clone()
        n_velocity_tensor_copy = n_velocity_tensor.clone()

        with torch.no_grad():
             # 下一个动作和 log_pi
            _, policy_action, log_pi, _ = self.actor(n_state_tensor_copy, n_q_tensor_copy, n_velocity_tensor_copy)
            # Target Q 计算
            target_Q1, target_Q2 = self.critic_target(n_state_tensor_copy, n_q_tensor_copy, n_velocity_tensor_copy, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
        
        # 当前状态 Q 估计
        state_tensor_copy = state_tensor.clone()
        q_tensor_copy = q_tensor.clone()
        velocity_tensor_copy = velocity_tensor.clone()
        current_Q1, current_Q2 = self.critic(state_tensor_copy, q_tensor_copy, velocity_tensor_copy, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 可选：记录 loss
        if self.total_it % self.learning_time == 0:
            self.writer.add_scalar('loss/critic_loss_q_weight', critic_loss, self.epoch)

        # 更新网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state_tensor, q_tensor, velocity_tensor):
        """更新Actor网络，使用图像+Q值+速度输入"""
        # 复制张量，防止原始张量被修改
        state_tensor_copy = state_tensor.clone()
        q_tensor_copy = q_tensor.clone()
        velocity_tensor_copy = velocity_tensor.clone()

         # 前向传播 Actor
        mu, pi, log_pi, _ = self.actor(state_tensor_copy, q_tensor_copy, velocity_tensor_copy, detach=True)
        
        # Actor loss 需要通过 Critic 来估计 Q
        actor_Q1, actor_Q2 = self.critic(state_tensor_copy, q_tensor_copy, velocity_tensor_copy, pi, detach=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        # 计算 Actor loss
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # 记录日志
        if self.total_it % self.learning_time == 0:
            self.writer.add_scalar('loss/actor_loss_q_weight', actor_loss, self.epoch)
        
        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新策略温度 alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, batch_size):
        [state_tensor, q_tensor, velocity_tensor, action, n_state_tensor, n_q_tensor, n_velocity_tensor, reward, not_done] = self.samplefromreplayer(batch_size, self.velocity_rms)

        self.update_critic(state_tensor, q_tensor, velocity_tensor, action, reward, n_state_tensor, n_q_tensor, n_velocity_tensor, not_done)
        self.update_actor_and_alpha(state_tensor, q_tensor, velocity_tensor)

        if self.total_it % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.q1, self.critic_target.q1, self.tau)
            utils.soft_update_params(self.critic.q2, self.critic_target.q2, self.tau)

    def step(self, exp_list):
        for exp in exp_list:
            if exp is not None:
                [O_z, O_q, O_velocity, action, next_O_z, next_O_q, next_O_velocity, reward, not_done] = exp
                self.replayer.store(O_z, O_q, O_velocity, action, next_O_z, next_O_q, next_O_velocity, reward, not_done)

    def myupdate(self, batch_size, velocity_rms):
        # 获取经验更新的数据
        [state_tensor, q_tensor, velocity_tensor, action, n_state_tensor, n_q_tensor, n_velocity_tensor, reward, not_done] = self.samplefromreplayer(batch_size, velocity_rms)

        # 使用获取到的经验来更新网络
        self.update_critic(state_tensor, q_tensor, velocity_tensor, action, reward, n_state_tensor, n_q_tensor, n_velocity_tensor, not_done)
        self.update_actor_and_alpha(state_tensor, q_tensor, velocity_tensor)

        [state_tensor2, q_tensor2, velocity_tensor2, action2, n_state_tensor2, n_q_tensor2, n_velocity_tensor2, reward2, not_done2] = self.samplefromreplayer(batch_size, velocity_rms)

        self.update_critic(state_tensor2, q_tensor2, velocity_tensor2, action2, reward2, n_state_tensor2, n_q_tensor2, n_velocity_tensor2, not_done2)

        utils.soft_update_params(self.critic.q1, self.critic_target.q1, self.tau)
        utils.soft_update_params(self.critic.q2, self.critic_target.q2, self.tau)

        del state_tensor, q_tensor, velocity_tensor, action, n_state_tensor, n_q_tensor, n_velocity_tensor, reward, not_done
        del state_tensor2, q_tensor2, velocity_tensor2, action2, n_state_tensor2, n_q_tensor2, n_velocity_tensor2, reward2, not_done2

    def samplefromreplayer(self, batch_size, velocity_rms):
        O_z, O_q, O_velocity, action, next_O_z, next_O_q, next_O_velocity, reward, not_done = self.replayer.sample(batch_size)
        O_velocity = velocity_rms.normalize(np.asarray(O_velocity))
        next_O_velocity = velocity_rms.normalize(np.asarray(next_O_velocity))

        state_tensor = torch.FloatTensor(O_z).to(device)
        q_tensor = torch.FloatTensor(O_q).to(device)
        velocity_tensor = torch.FloatTensor(O_velocity).float().to(device)
        n_state_tensor = torch.FloatTensor(next_O_z).to(device)
        n_q_tensor = torch.FloatTensor(next_O_q).to(device)
        n_velocity_tensor = torch.FloatTensor(next_O_velocity).float().to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(device)

        return [state_tensor, q_tensor, velocity_tensor, action, n_state_tensor, n_q_tensor, n_velocity_tensor, reward, not_done]

    def learn(self, velocity_rms):
        if self.replayer.count > self.batch_size:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f'training!!! {formatted_now}')
            for count in range(self.learning_time):
                self.total_it += 1
                self.myupdate(self.batch_size, velocity_rms)

    def save(self, epoch, policy_path):
        actor_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }
        critic_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(actor_checkpoint, policy_path + '/actor_{:03d}'.format(epoch))
        torch.save(critic_checkpoint, policy_path + '/critic_{:03d}'.format(epoch))

    def load(self, model_file, mode, policy_epoch_index=0):
        actor_file = model_file + '/actor_' + "{:03}".format(policy_epoch_index)
        critic_file = model_file + '/critic_' + "{:03}".format(policy_epoch_index)
        # decoder_file = model_file + '/decoder_' + "{:03}".format(policy_epoch_index)

        # 加载断点模型
        actor_state = torch.load(actor_file)
        critic_state = torch.load(critic_file)
        # decoder_state = torch.load(decoder_file)
        
        # 加载断点的状态
        self.actor.load_state_dict(actor_state['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_state['optimizer_state_dict'])
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(critic_state['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_state['optimizer_state_dict'])
        self.critic_target = copy.deepcopy(self.critic)

        starting_epoch = actor_state['epoch'] + 1
        self.epoch = actor_state['epoch']

        if mode == 'test':
            self.actor.eval()

        return starting_epoch
