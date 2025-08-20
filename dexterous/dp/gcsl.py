




def gcsl():
    dataset = load_from_hdf5(dataset_path)
    policy = load_from_cfg(policy_cfg)
    goal_sampler = load_from_cfg(goal_cfg)

    for i in range(n_iters):
        policy = train(policy, dataset, goal_mode="relabel")
        
        rollouts, performance = rollout(policy, goal_sampler)
        dataset = dataset.add(rollouts)

    return policy


def her():
    """
    maybe more sensitive to hyperparameters because it uses rl (argued by gcsl paper)
    """
    dataset = load_from_hdf5(dataset_path)

    policy = load_from_cfg(policy_cfg)
    goal_sampler = load_from_cfg(goal_cfg)
    
    replay_buffer = goal_relabel(dataset)

    for i in range(n_iters):
        rollouts = rollout_noisy(actor, goal_sampler, noise=eps, n=n_updates/utd)
        new_goals = goal_relabel(rollouts)
        replay_buffer = replay_buffer.add(new_goals)

        for j in range(n_updates):
            update(policy, replay_buffer)

    return policy


def ddpg_update(policy, replay_buffer):
    actor, critic, actor_target, critic_target = policy

    batch = replay_buffer.sample()
    states,actions,rewards = batch

    critic_targets = rewards + critic_target(next_states, actor_target(next_states))
    critic_loss = sg(critic_targets) - critic(states, actions)
    critic_loss.backward()

    actor_loss = critic(states, actor(states))
    actor_loss.backward()

    critic_target = ema(critic, critic_target, rho)
    actor_target = ema(actor, actor_target, rho)
