import os
import random
import time
from dataclasses import dataclass
import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils import make_env_discrete
import nets.ppo_nets as nets
from trainers.trainers_ppo import train


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "slow-agent"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Breakout-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    N_hidden_layers: int = 4
    """the number of hidden layers in the policy and value networks"""
    agent: str = None
    """the agent class to use"""
    repeat_frame: int = 1
    """repeat each obs for this number of frames for the agent"""
    remove_skip: bool = False
    """remove skip frame wrapper in atairi env"""
    save_model: bool = False
    """save the model"""
    eval_model: bool = True
    """evaluate the model"""
    frame_skip: int = 1
    """how many frames to skip while processing one layer of NN; speed of env = frame_skip * speed of neuron firing"""
    get_instant_activations: bool = True
    """get nonzero initial activation with an instance forward pass of the agent"""
    history_states: int = 1
    """number of history states to consider"""
    num_last_actions: int = 0
    """number of last actions to consider"""
    add_last_action: bool = False
    """add last action to the input in fully connected layers"""
    actor_hidden_dim: int = 256
    """the hidden dimension of the actor network in case of MLP actor"""
    wd: float = 0.0
    """weight decay"""
    log_interval: int = 1000
    """log interval"""
    episodic_return_threshold: int = 10000
    """episodic return threshold"""
    n_fc_layers: int = 2
    """number of fully connected layers in the head of the actor"""
    lstm: bool = False
    """whether to use LSTM in the actor"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.mininum_steps_size = int(args.num_steps // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print('\n args.batch_size', args.batch_size, 'args.minibatch_size', args.minibatch_size,
          'args.mininum_steps_size', args.mininum_steps_size, 'args.num_iterations', args.num_iterations, '\n')
    group_name = f"{args.agent}_fs{args.frame_skip}_rf{args.repeat_frame}_hs{args.history_states}_ha{args.num_last_actions}_{args.exp_name}"
    run_name = f"{group_name}_s{args.seed}_{int(time.time())}"
    args.group_name = group_name
    args.run_name = run_name
    args.randint = random.randint(0, 1000)
    args.out_dir = f"runs/{args.env_id}/{args.group_name}/{run_name}_randint{args.randint}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            group=group_name,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=["iclr"]
        )
    writer = SummaryWriter(args.out_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    with open(f'{args.out_dir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_discrete(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # filling in missing arguments from environment
    if 'MinAtar' in args.env_id or 'MiniGrid' in args.env_id:
        args.input_channels = envs.single_observation_space.shape[-1]
    else:
        args.input_channels = envs.single_observation_space.shape[0]
    args.obs_shape = envs.single_observation_space.shape

    Agent = getattr(nets, args.agent)
    agent = Agent(envs, args).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.wd)

    train(args, optimizer, device, envs, agent, writer)

    envs.close()
    writer.close()