# Official Pytorch Implementation of the paper [Handling Delay in Real-Time Reinforcement Learning](https://openreview.net/forum?id=YOc5t8PHf2)

# Installation

For Mujoco the code was tested with python3.10.

Download Mujoco from [here](https://github.com/google-deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz) 
and place it in /home/$USER/.mujoco, and then source it:

```
export MUJOCO_PLUGIN_PATH=/home/$USER/.mujoco/mujoco-2.3.3/bin/mujoco_plugin/
```

and 

```
export MUJOCO_PATH=/home/$USER/.mujoco/mujoco-2.3.3/
```

Install dependencies with:

```
pip install -r requirements.txt
```

# Tranning Mujoco

Vanilla SAC is taken from [cleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

To train the vanilla SAC algorithm without delay run:

```
python train_sac.py --env_id HalfCheetah-v4 --agent Actor
```

To train the agent without any skip connections within a parallel computation framework and neuron execution time of 1 run:

```
python train_sac.py --env_id HalfCheetah-v4 --trainer delayed --agent ActorSlow --frame_skip 1 
```

To train the agent with skip connections and with state-augmentation within a parallel computation framework and neuron execution time of 1 run:

```
python train_sac.py --env_id HalfCheetah-v4 --trainer delayed --agent ActorSlowConcat --num_last_actions 2 --frame_skip 1 
```

To train it with different neuron execution times change the `frame_skip` parameter to 2,3 or 4. 

# Tranning MinAtar and MiniGrid
For MinAtar and MiniGrid the code was tested with python3.9.

Install MinAtar using the instructions [here](https://github.com/kenjyoung/MinAtar).

For training Vanilla PPO without delay (heavily based on [cleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)), run: 
```
python train_ppo.py --env_id MinAtar/Breakout-v0 --agent AgentSeparateActorCritic
```

For training an agent without skip connections within a parallel computation framework and neuron execution time of 1 run:

```
python train_ppo.py --env_id MinAtar/Breakout-v0 --agent ActorSlowPPO --frame_skip 1 
```

For training an agent with skip connections and with state-augmentation within a parallel computation framework and neuron execution time of 1 run

for MinAtar:
```
python train_ppo.py --env_id MinAtar/Breakout-v0 --agent ActorSlowSkipResPPO --add_last_action --frame_skip 1 
```

and for MiniGrid:
```
python train_ppo.py --env_id MiniGrid-DoorKey-5x5-v0 --agent ActorSlowSkipResPPO --history_states 4 --frame_skip 1 
```
