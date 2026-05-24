import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class MlpPolicy(nn.Module):
    def __init__(self, action_size, input_size, hidden_dim=512, hidden_layers=2):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        layers = []
        last_dim = self.input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.fc_pi = nn.Linear(hidden_dim, self.action_size)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def pi(self, x):
        x = self.trunk(x)
        x = self.fc_pi(x)
        return self.softmax(x)

    def v(self, x):
        x = self.trunk(x)
        x = self.fc_v(x)
        return x
