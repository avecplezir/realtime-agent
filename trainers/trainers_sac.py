import time
import torch
import numpy as np
import torch.nn.functional as F


def train(args, envs, actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, a_optimizer, rb, writer, device, alpha, log_alpha, target_entropy):
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    actor_updates = 0

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actor.eval()
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actor.train()
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                actor.eval()
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                actor.train()
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_updates += 1
                    actor_optimizer.zero_grad()
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_loss.backward()
                    actor.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            actor.eval()
                            _, log_pi, _ = actor.get_action(data.observations)
                            actor.train()
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.log_interval == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("actor_updates", actor_updates, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if args.save_model:
        model_path = f"{args.out_dir}/model"
        print('Saving model to', model_path)
        torch.save(actor.state_dict(), model_path)

    if args.eval_model:
        print('evaluating model after training')
        from utils import evaluate_sac
        episodic_returns = evaluate_sac(actor, args, args.device)
        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_episodic_return", mean)
        writer.add_scalar("eval/std_episodic_return", std)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        print('evaluating model after training using mean')
        episodic_returns = evaluate_sac(actor, args, args.device, greedy=True)

        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_mode_episodic_return", mean)
        writer.add_scalar("eval/std_mode_episodic_return", std)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/mode_episodic_return", episodic_return, idx)

    envs.close()
    writer.close()


def train_delayed(args, envs, actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, a_optimizer, rb, writer, device, alpha, log_alpha, target_entropy):
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    actor_updates = 0
    global_step_threshold = 0
    observations = torch.Tensor(obs).to(device)
    hidden_activations = actor.get_activations(observations)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actor.eval()
            with torch.no_grad():
                actions, _, _, hidden_activations = actor.get_action(observations, hidden_activations)
            actor.train()
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    if global_step > global_step_threshold:
                        print( f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}" )
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        global_step_threshold += args.episodic_return_threshold

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add((obs, real_next_obs, actions, rewards, terminations))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        observations = torch.Tensor(next_obs).to(device)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample_seq(args.N_hidden_layers + 2 + 1 + args.warm_up_seq, args.batch_size) # (args.N_hidden_layers + 2) is actual number of layer in NN, +1 is for the delay, +args.warm_up_seq is for the warm up sequence
            with torch.no_grad():
                actor.eval()
                next_state_actions, next_state_log_pi, _ = actor.learn_action(data.next_observations)
                actor.train()
                qf1_next_target = qf1_target(data.next_observations[-1], next_state_actions)
                qf2_next_target = qf2_target(data.next_observations[-1], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards[-1].flatten() + (1 - data.dones[-1].flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations[-1], data.actions[-1]).view(-1)
            qf2_a_values = qf2(data.observations[-1], data.actions[-1]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_iteration
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_updates += 1
                    actor_optimizer.zero_grad()
                    pi, log_pi, _ = actor.learn_action(data.observations)
                    qf1_pi = qf1(data.observations[-1], pi)
                    qf2_pi = qf2(data.observations[-1], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_loss.backward()
                    actor.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            actor.eval()
                            _, log_pi, _ = actor.learn_action(data.observations)
                            actor.train()
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.log_interval == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("actor_updates", actor_updates, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if args.save_model:
        model_path = f"{args.out_dir}/model"
        print('Saving model to', model_path)
        torch.save(actor.state_dict(), model_path)

    if args.eval_model:
        print('evaluating model after training')
        from utils import evaluate_sac_delayed
        print('evaluating model after training using mean')

        mode_episodic_returns = evaluate_sac_delayed(actor, args, args.device, greedy=True)
        mode_episodic_returns = np.array(mode_episodic_returns)
        mode_mean, mode_std = np.mean(mode_episodic_returns), np.std(mode_episodic_returns)
        writer.add_scalar("eval/mean_mode_episodic_return", mode_mean)
        writer.add_scalar("eval/std_mode_episodic_return", mode_std)

        for idx, episodic_return in enumerate(mode_episodic_returns):
            writer.add_scalar("eval/mode_episodic_return", episodic_return, idx)

        print('evaluating model after training using sampling')
        episodic_returns = evaluate_sac_delayed(actor, args, args.device)
        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_episodic_return", mean)
        writer.add_scalar("eval/std_episodic_return", std)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
