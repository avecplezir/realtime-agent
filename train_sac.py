# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import copy
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
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from utils import make_env_continuous
from nets.sac_nets import SoftQNetwork
import trainers
from replay_buffer import ReplayMemory

@dataclass
class Args:
    exp_name: str = ""
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
    env_id: str = "Pendulum"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 10000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    update_frequency: int = 4
    """the frequency of training updates"""
    policy_iteration: int = -1
    """how many times to update the actor per iteration"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    frame_skip: int = 1
    """how many frames to skip while processing one layer of NN; speed of env = frame_skip * speed of neuron firing"""
    agent: str = 'default'
    """type of the agent, options: default, wm, wm_delayed"""
    num_envs: int = 1
    """number of envs to collect experiance"""
    N_hidden_layers: int = 1
    """number of hidden layers in Agent"""
    actor_hidden_dim: int = 256
    """number of neurons in hidden layers of the actor"""
    trainer: str = 'default'
    """type of the trainer"""
    warm_up_seq: int = 0
    """warm up sequence length for the actor while trainig with delayed action sampling from the buffer"""
    history_states: int = 1
    """number of history states to include in the observation"""
    num_last_actions: int = 0
    """number of last actions to include in the observation"""
    target_entropy_scale: float = 1.
    """coefficient for scaling the autotune entropy target"""
    save_model: bool = True
    """save the actor model"""
    eval_model: bool = True
    """evaluate the model"""
    capture_video_eval: bool = False
    """capture video during evaluation"""
    get_instant_activations: bool = False
    """get nonzero initial activation with an instance forward pass of the agent"""
    max_steps: int = 1000
    """max steps for a custom mujoco env"""
    speed_up: int = 1
    """speed up the environment, make the time between observations smaller / speed_up"""
    wd: float = 0.
    """weight decay"""
    log_interval: int = 1000
    """log interval"""
    episodic_return_threshold: int = 1000
    """episodic return threshold"""

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    args.policy_iteration = args.policy_frequency if args.policy_iteration < 1 else args.policy_iteration
    print('args', args)

    group_name = f"{args.agent}_fs{args.frame_skip}_N{args.N_hidden_layers+2}_ha{args.num_last_actions}_hs{args.history_states}"
    if args.exp_name != "":
        group_name = f"{group_name}_{args.exp_name}"
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
    envs = gym.vector.SyncVectorEnv([make_env_continuous(args.env_id, args.seed, 0, args.capture_video, run_name, args) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    args.initial_obs_size = int((obs_dim - args.num_last_actions * np.array(envs.single_action_space.shape).prod()) / args.history_states)

    max_action = float(envs.single_action_space.high[0])

    try:
        import nets
        Actor = getattr(nets.sac_nets, args.agent)
        actor = Actor(envs, args).to(device)
    except AttributeError:
        raise ValueError(f"unknown agent type {args.agent}")
    print('actor', actor)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.wd)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        a_optimizer = None
        log_alpha = None
        target_entropy = None

    envs.single_observation_space.dtype = np.float32

    os.environ['MUJOCO_GL'] = "egl"

    if args.trainer == 'default':
        from trainers.trainers_sac import train
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
        print('default sac training')
        train(args, envs, actor, qf1, qf2, qf1_target, qf2_target,
                      q_optimizer, actor_optimizer, a_optimizer, rb, writer, device, alpha, log_alpha, target_entropy)

    elif args.trainer == 'delayed':
        from trainers.trainers_sac import train_delayed
        rb = ReplayMemory(
            args.buffer_size,
            np.array(envs.single_observation_space.shape).prod(),
            np.array(envs.single_action_space.shape).prod(),
            envs.single_observation_space.dtype,
            device=device,
        )
        print('starting delayed SAC buffer training loop')
        train_delayed(args, envs, actor, qf1, qf2, qf1_target, qf2_target,
                      q_optimizer, actor_optimizer, a_optimizer,
                      rb, writer, device, alpha, log_alpha, target_entropy)

    else:
        assert 0, 'Unknown trainer'
