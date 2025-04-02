import torch.nn as nn 
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import torch.nn.functional as F


def transform_obs(x, env_id):
    if 'MinAtar' in env_id:
        x = x.permute(0, 3, 1, 2).float()
    elif 'MiniGrid' in env_id:
        x = x.permute(0, 3, 1, 2).float() / 255.0
    else:
        x = x / 255.0
    return x


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        self.args = args
        if 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.network = nn.Sequential(
                nn.Sequential(layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                              nn.ReLU()),
                nn.Flatten(),
                layer_init(nn.Linear(576, 512)),
            )
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)

        elif 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.network = nn.Sequential(
                nn.Sequential(layer_init(nn.Conv2d(obs_shape[-1], 16, 3, padding='same')),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(16, 32, 3,  padding='same')),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(32, 32, 3, padding='same')),
                              nn.ReLU()),
                nn.Flatten(),
                layer_init(nn.Linear(3200, 512)),
            )
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)

        else:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x, *args, **kwargs):
        x = transform_obs(x, self.args.env_id)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, hidden_acts=None, action=None):
        x = transform_obs(x, self.args.env_id)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), []

    def get_activations(self, x):
        return []


def get_critic(args, envs):
    if 'MiniGrid' in args.env_id:
        obs_shape = envs.single_observation_space.shape
        critic = nn.Sequential(
            nn.Sequential(layer_init(nn.Conv2d(args.input_channels, 32, 8, stride=4)),
                          nn.ReLU()),
            nn.Sequential(layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                          nn.ReLU()),
            nn.Sequential(layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                          nn.ReLU()),
            nn.Sequential(nn.Flatten(),
                          layer_init(nn.Linear(576, 512)),
                          nn.ReLU()),
            layer_init(nn.Linear(512, 1), std=1)
        )
    elif 'MinAtar' in args.env_id:
        critic = nn.Sequential(
            layer_init(nn.Conv2d(args.input_channels, 16, 3, padding='same')),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, padding='same')),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding='same')),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3200, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )
    else:
        critic = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )
    return critic


class AgentSeparateActorCritic(Agent):
    def __init__(self, envs, args):
        nn.Module.__init__(self)

        self.args = args
        self.critic = get_critic(args, envs)
        if 'MiniGrid' in self.args.env_id:
            self.actor = nn.Sequential(
                nn.Sequential(layer_init(nn.Conv2d(args.input_channels, 32, 8, stride=4)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                              nn.ReLU()),
                nn.Sequential(nn.Flatten(),
                              layer_init(nn.Linear(576, 512)),
                              nn.ReLU()),
                layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            )
        elif 'MinAtar' in self.args.env_id:
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(args.input_channels, 16, 3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, 3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 32, 3, padding='same')),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(3200, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            )

    def get_action_and_value(self, x, hidden_acts=None, action=None):
        x = transform_obs(x, self.args.env_id)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), []

    def get_value(self, x, *args, **kwargs):
        x = transform_obs(x, self.args.env_id)
        return self.critic(x)


class ActorSlowPPO(AgentSeparateActorCritic):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic = get_critic(args, envs)
        self.add_last_action = args.add_last_action

        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
            )

            with torch.no_grad():
                dem = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv[:-1]):
                    dem = block(dem)
                    shape = np.array(dem.shape).prod()

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )
            with torch.no_grad():
                dem = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv):
                    dem = block(dem)
                    shape = np.array(dem.shape).prod()

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )

            with torch.no_grad():
                dem = torch.zeros((1, *obs_shape))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv):
                    dem = block(dem)
                    shape = np.array(dem.shape).prod()

        add_last_action_dim = envs.single_action_space.n if self.add_last_action else 0
        self.action_dim = envs.single_action_space.n
        if args.lstm:
            self.lstm = nn.LSTMCell(shape + add_last_action_dim, 512)
            self.fc_logits = layer_init(nn.Linear(512 + add_last_action_dim, envs.single_action_space.n))
        else:
            self.fc = nn.ModuleList()
            if args.n_fc_layers == 2:
                self.fc.append(nn.Sequential(layer_init(nn.Linear(shape + add_last_action_dim, 512)), nn.ReLU()))
                self.fc.append(nn.Linear(512 + add_last_action_dim, envs.single_action_space.n))
            elif args.n_fc_layers == 1:
                self.fc.append(layer_init(nn.Linear(shape + add_last_action_dim, envs.single_action_space.n)))

    def init_activation(self, x):
        if self.args.get_instant_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_activations(self, x):
        x = transform_obs(x, self.args.env_id)

        hidden_activations = []
        for block in list(self.conv):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        x = nn.Flatten()(x)

        if self.args.lstm:
            lstm_state = (
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
            )
            if self.add_last_action:
                last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                x = torch.cat([x, last_action], dim=1)
            h, c = self.lstm(x, lstm_state)
            hidden_activations.append((self.init_activation(h), self.init_activation(c)))
            x = h
            if self.add_last_action:
                last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                x = torch.cat([x, last_action], dim=1)
            x = self.fc_logits(x)
            hidden_activations.append(self.init_activation(x))
        else:
            for fc in self.fc:
                if self.add_last_action:
                    last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                    x = torch.cat([x, last_action], dim=1)
                x = fc(x)
                hidden_activations.append(self.init_activation(x))

        if self.add_last_action:
            policy_dist = Categorical(logits=x)
            action = policy_dist.sample()
            hidden_activations.append(self.init_activation(action))

        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []
        for input, block in zip([x] + hidden_acts[:2], list(self.conv)):
            out = F.relu(block(input))
            new_hidden.append(out)

        if self.args.lstm:
            input = hidden_acts[2]
            input = nn.Flatten()(input)
            hs = hidden_acts[3]
            if self.add_last_action:
                last_action = hidden_acts[-1]
                last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                input = torch.cat([input, last_action], dim=1)
            lstm_state = self.lstm(input, hs)
            new_hidden.append(lstm_state)
            input = hidden_acts[3][0]
            if self.add_last_action:
                last_action = hidden_acts[-1]
                last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                input = torch.cat([input, last_action], dim=1)
            new_logits = self.fc_logits(input)
            new_hidden.append(new_logits)
        else:
            for input, fc in zip(hidden_acts[2:], self.fc):
                input = nn.Flatten()(input)
                if self.add_last_action:
                    last_action = hidden_acts[-1]
                    last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                    input = torch.cat([input, last_action], dim=1)
                out = fc(input)
                new_hidden.append(out)

        logits = hidden_acts[-2] if self.add_last_action else hidden_acts[-1]
        return logits, new_hidden

    def get_action_and_value(self, x, hidden_acts=None, action=None):
        x = transform_obs(x, self.args.env_id)
        logits, new_hidden = self(x, hidden_acts)
        policy_dist = Categorical(logits=logits)
        if action is None:
            action = policy_dist.sample()
        if self.add_last_action:
            new_hidden.append(action)
        return action, policy_dist.log_prob(action), policy_dist.entropy(), self.critic(x), new_hidden


class ActorSlowSkipPPO(ActorSlowPPO):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic = get_critic(args, envs)

        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
            )

            with torch.no_grad():
                dem = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv):
                    dem = block(dem)
                    shape += np.array(dem.shape).prod()

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )

            with torch.no_grad():
                dem = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                outs = []
                for block in list(self.conv):
                    dem = block(dem)
                    outs.append(dem)

                spatial = outs[-1].shape[-1]
                shape = np.array([1, obs_shape[-1], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1] // spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape += np.array(out.shape).prod()

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )

            with torch.no_grad():
                dem = torch.zeros((1, *obs_shape))
                outs = []
                for block in list(self.conv):
                    dem = block(dem)
                    outs.append(dem)

                spatial = outs[-1].shape[-1]
                shape = np.array([1, obs_shape[0], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1] // spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape += np.array(out.shape).prod()

        if args.lstm:
            self.lstm = nn.LSTMCell(shape, 512)
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))
        else:
            self.fc = nn.ModuleList()
            if args.n_fc_layers == 2:
                self.fc.append(nn.Sequential(layer_init(nn.Linear(shape, 512)), nn.ReLU()))
                self.fc.append(layer_init(nn.Linear(512, envs.single_action_space.n)))
            elif args.n_fc_layers == 1:
                self.fc.append(layer_init(nn.Linear(shape, envs.single_action_space.n)))

    def get_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        orig_x = x
        hidden_activations = []
        for block in list(self.conv):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        reshaped = []
        all_items = [orig_x] + hidden_activations
        last_spatial_size = all_items[-1].shape[-1]
        for item in all_items:
            max_pool_size = item.shape[-1]//last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))

        x = torch.cat(reshaped, dim=1)
        if self.args.lstm:
            lstm_state = (
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
            )
            h, c = self.lstm(x, lstm_state)
            hidden_activations.append((self.init_activation(h), self.init_activation(c)))
            logits = self.fc_logits(h)
            hidden_activations.append(self.init_activation(logits))
        else:
            for fc in self.fc:
                x = fc(x)
                hidden_activations.append(self.init_activation(x))

        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []
        for i, (input, block) in enumerate(zip([x] + hidden_acts[:2], list(self.conv))):
            out = F.relu(block(input))
            new_hidden.append(out)

        reshaped = []
        last_spatial_size = hidden_acts[2].shape[-1]
        for item in ([x] + hidden_acts[:3]):
            max_pool_size = item.shape[-1] // last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))

        if self.args.lstm:
                input = torch.cat(reshaped, dim=1)
                input = nn.Flatten()(input)
                hs = hidden_acts[-2]
                lstm_state = self.lstm(input, hs)
                new_hidden.append(lstm_state)
                new_logits = self.fc_logits(hidden_acts[-2][0])
                new_hidden.append(new_logits)
        else:
            for i, (input, fc) in enumerate(zip(hidden_acts[2:], self.fc)):
                if i == 0:
                    input = torch.cat(reshaped, dim=1)
                    input = nn.Flatten()(input)
                out = fc(input)
                new_hidden.append(out)
        logits = hidden_acts[-1]

        return logits, new_hidden


class ActorSlowSkipResPPO(ActorSlowPPO):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic = get_critic(args, envs)
        self.add_last_action = args.add_last_action

        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32 + 32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64 + 64, 64, kernel_size=3, padding='same')),
            )

            with torch.no_grad():
                x = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(x.shape).prod()
                res = torch.zeros_like(x)
                for i, block in enumerate(list(self.conv)):
                    if i == 0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])], 1)))
                        res = x
                    shape += np.array(x.shape).prod()

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32 + 32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64 + 64, 64, 3, stride=1)),
            )

            with torch.no_grad():
                x = torch.zeros((1, obs_shape[-1], obs_shape[0], obs_shape[1]))
                outs = []
                res = torch.zeros_like(x)
                for i, block in enumerate(list(self.conv)):
                    if i == 0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])], 1)))
                        res = x
                    outs.append(torch.zeros_like(x))

                spatial = outs[-1].shape[-1]
                shape = np.array([1, obs_shape[-1], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1] // spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape += np.array(out.shape).prod()

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32 + 32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1)),
            )

            with torch.no_grad():
                x = torch.zeros((1, *obs_shape))
                outs = []
                res = torch.zeros_like(x)
                for i, block in enumerate(list(self.conv)):
                    if i == 0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])], 1)))
                        res = x
                    outs.append(torch.zeros_like(x))

                spatial = outs[-1].shape[-1]
                shape = np.array([1, obs_shape[0], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1] // spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape += np.array(out.shape).prod()

        add_last_action_dim = envs.single_action_space.n if self.add_last_action else 0
        self.action_dim = envs.single_action_space.n
        if args.lstm:
            self.lstm = nn.LSTMCell(shape + add_last_action_dim, 512)
            self.fc_logits = layer_init(nn.Linear(512 + add_last_action_dim, envs.single_action_space.n))
        else:
            self.fc = nn.ModuleList()
            if args.n_fc_layers == 2:
                self.fc.append(nn.Sequential(layer_init(nn.Linear(shape + add_last_action_dim, 512)), nn.ReLU()))
                self.fc.append(layer_init(nn.Linear(512 + add_last_action_dim, envs.single_action_space.n)))
            elif args.n_fc_layers == 1:
                self.fc.append(layer_init(nn.Linear(shape + add_last_action_dim, envs.single_action_space.n)))

    def get_activations(self, x):
        x = transform_obs(x, self.args.env_id)

        orig_x = x
        hidden_activations = []

        res = torch.zeros_like(x)
        for i, block in enumerate(list(self.conv)):
            if i == 0:
                x = F.relu(block(x))
                res = x
            else:
                x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])], 1)))
                res = x
            hidden_activations.append(self.init_activation(x))

        reshaped = []
        all_items = [orig_x] + hidden_activations
        last_spatial_size = all_items[-1].shape[-1]
        for item in all_items:
            max_pool_size = item.shape[-1] // last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))

        x = torch.cat(reshaped, dim=1)
        if self.args.lstm:
            lstm_state = (
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
            )
            if self.add_last_action:
                last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                x = torch.cat([x, last_action], dim=1)
            h, c = self.lstm(x, lstm_state)
            hidden_activations.append((self.init_activation(h), self.init_activation(c)))
            x = h
            if self.add_last_action:
                last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                x = torch.cat([x, last_action], dim=1)
            x = self.fc_logits(x)
            hidden_activations.append(self.init_activation(x))
        else:
            for fc in self.fc:
                if self.add_last_action:
                    last_action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
                    x = torch.cat([x, last_action], dim=1)
                x = fc(x)
                hidden_activations.append(self.init_activation(x))

        if self.add_last_action:
            policy_dist = Categorical(logits=x)
            action = policy_dist.sample()
            hidden_activations.append(self.init_activation(action))

        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []
        res = torch.zeros_like(x)
        for i, (input, block) in enumerate(zip([x] + hidden_acts[:2], list(self.conv))):
            if i == 0:
                out = F.relu(block(x))
                res = out
            else:
                out = F.relu(block(torch.cat([input, F.interpolate(res, out.shape[-1])], 1)))
                res = out
            new_hidden.append(out)

        reshaped = []
        last_spatial_size = hidden_acts[2].shape[-1]
        for item in ([x] + hidden_acts[:3]):
            max_pool_size = item.shape[-1] // last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))

        if self.args.lstm:
            input = torch.cat(reshaped, dim=1)
            input = nn.Flatten()(input)
            hs = hidden_acts[3]

            if self.add_last_action:
                last_action = hidden_acts[-1]
                last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                input = torch.cat([input, last_action], dim=1)

            lstm_state = self.lstm(input, hs)
            new_hidden.append(lstm_state)

            input = hidden_acts[3][0]
            if self.add_last_action:
                last_action = hidden_acts[-1]
                last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                input = torch.cat([input, last_action], dim=1)

            new_logits = self.fc_logits(input)
            new_hidden.append(new_logits)
        else:
            for i, (input, fc) in enumerate(zip(hidden_acts[2:], self.fc)):
                if i == 0:
                    input = torch.cat(reshaped, dim=1)
                    input = nn.Flatten()(input)

                if self.add_last_action:
                    last_action = hidden_acts[-1]
                    last_action = torch.eye(self.action_dim).to(last_action.device)[last_action]
                    input = torch.cat([input, last_action], dim=1)

                out = fc(input)
                new_hidden.append(out)

        logits = hidden_acts[-2] if self.add_last_action else hidden_acts[-1]

        return logits, new_hidden


class ActorSlowConcatPPO(ActorSlowPPO):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic = get_critic(args, envs)

        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32 + obs_shape[-1], 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64 + obs_shape[-1], 64, kernel_size=3, padding='same')),
            )
            shape = 6400

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32 + obs_shape[-1], 64, 4, stride=2)),
                layer_init(nn.Conv2d(64 + obs_shape[-1], 64, 3, stride=1)),
            )
            shape = 576

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32 + obs_shape[0], 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64 + obs_shape[0], 64, kernel_size=3, stride=1)),
            )
            shape = 3136

        if args.lstm:
            self.lstm = nn.LSTMCell(shape, 512)
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))
        else:
            self.fc = nn.ModuleList()
            if args.n_fc_layers == 2:
                self.fc.append(nn.Sequential(layer_init(nn.Linear(shape, 512)), nn.ReLU()))
                self.fc.append(layer_init(nn.Linear(512, envs.single_action_space.n)))
            elif args.n_fc_layers == 1:
                self.fc.append(layer_init(nn.Linear(shape, envs.single_action_space.n)))

    def get_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        obs = x

        hidden_activations = []
        for i, block in enumerate(list(self.conv[:3])):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))
            if i < 2:
                obs_int = F.interpolate(obs, (x.shape[2], x.shape[3]))
                x = torch.cat([obs_int, x], dim=1)
        x = nn.Flatten()(x)

        if self.args.lstm:
            lstm_state = (
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
                torch.zeros(x.shape[0], self.lstm.hidden_size).to(x.device),
            )
            h, c = self.lstm(x, lstm_state)
            hidden_activations.append((self.init_activation(h), self.init_activation(c)))
            logits = self.fc_logits(h)
            hidden_activations.append(self.init_activation(logits))
        else:
            for fc in self.fc:
                x = fc(x)
                hidden_activations.append(self.init_activation(x))

        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []

        for i, (input, block) in enumerate(zip([x] + hidden_acts[:2], list(self.conv)[:3])):
            if i != 0:
                obs_int = F.interpolate(x, (input.shape[2], input.shape[3]))
                input = torch.cat([obs_int, input], dim=1)
            out = F.relu(block(input))
            new_hidden.append(out)

        if self.args.lstm:
            input = nn.Flatten()(hidden_acts[-3])
            hs = hidden_acts[-2]
            lstm_state = self.lstm(input, hs)
            new_hidden.append(lstm_state)
            new_logits = self.fc_logits(hidden_acts[-2][0])
            new_hidden.append(new_logits)
        else:
            for i, (input, fc) in enumerate(zip(hidden_acts[2:], self.fc)):
                input = nn.Flatten()(input)
                out = fc(input)
                new_hidden.append(out)

        logits = hidden_acts[-1]
        return logits, new_hidden

