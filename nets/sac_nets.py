import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        bias = True
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape), bias=bias)
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape), bias=bias)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        for block in self.blocks:
            x = F.relu(block(x))
        if self.args.train_only_last_layer:
            x = x.detach()
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def sample_action(self, mean, log_std):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, x):
        mean, log_std = self(x)
        action, log_prob, mean = self.sample_action(mean, log_std)
        return action, log_prob, mean

    def backward(self, *args):
        pass


class ActorSlow(Actor):
    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        new_mean = self.fc_mean(hidden_activations[-2])
        new_log_std = self.fc_logstd(hidden_activations[-2])
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_action(self, x, hidden_activations):
        mean, log_std, hidden_activations = self(x, hidden_activations)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if self.args.trainer == 'delayed_sampled':
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, torch.exp(log_prob), hidden_activations
        else:
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, mean, hidden_activations

    def init_activation(self, x):
        if self.args.get_instant_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_activations(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations

    def learn_action(self, obs):
        hidden_activations = self.get_activations(obs[0])
        for i, ob in enumerate(obs):
            action, log_prob, mean, hidden_activations = self.get_action(ob, hidden_activations)
        return action, log_prob, mean


class ActorSlowSkip(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(self.input_dim + (args.N_hidden_layers+1)*hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(self.input_dim + (args.N_hidden_layers+1)*hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        cat_hidden = torch.cat([obs] + hidden_activations[:-1], dim=1)
        new_mean = self.fc_mean(cat_hidden)
        new_log_std = self.fc_logstd(cat_hidden)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x,  last_action=None, last_reward=None):
        obs = x
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        x = torch.cat([obs] + hidden_activations, dim=1)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations


class ActorSlowConcat(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(self.input_dim + hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(self.input_dim + hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(self.input_dim + hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        out = F.relu(self.blocks[0](obs))
        new_hidden_activations.append(out)
        for input, block in zip(hidden_activations[:-2], self.blocks[1:]):
            input = torch.cat([obs, input], dim=1)
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        input = torch.cat([obs, hidden_activations[-2]], dim=1)
        new_mean = self.fc_mean(input)
        new_log_std = self.fc_logstd(input)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x,  last_action=None, last_reward=None):
        obs = x
        hidden_activations = []
        x = F.relu(self.blocks[0](x))
        hidden_activations.append(self.init_activation(x))
        for block in self.blocks[1:]:
            x = torch.cat([obs, x], dim=1)
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        x = torch.cat([obs, x], dim=1)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations


class ActorSlowResSkip(ActorSlowSkip):
    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        res = 0
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append((out+res)/np.sqrt(2))
            res = out
        cat_hidden = torch.cat([obs] + hidden_activations[:-1], dim=1)
        new_mean = self.fc_mean(cat_hidden)
        new_log_std = self.fc_logstd(cat_hidden)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations
