import numpy as np

def simulate(agent, envs, steps=0, episodes=0, state=None):
	"""

	:param agent	: 算法类agent
	:param envs	: 环境集合
	:param steps	:
	:param episodes:
	:param state	:
	:return:
	"""
	# Initialize or unpack simulation state.
	if state is None:
		step, episode = 0, 0
		done = np.ones(len(envs), np.bool)
		length = np.zeros(len(envs), np.int32)
		obs = [None] * len(envs)
		agent_state = None
	else:
		step, episode, done, length, obs, agent_state = state
	while (steps and step < steps) or (episodes and episode < episodes):
		# Reset envs if necessary.
		if done.any():		# any 为或操作
			indices = [index for index, d in enumerate(done) if d]
			promises = [envs[i].reset(blocking=False) for i in indices]
			for index, promise in zip(indices, promises):
				obs[index] = promise()
		# Step agents.
		obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
		action, agent_state = agent(obs, done, agent_state)
		action = np.array(action)
		assert len(action) == len(envs)
		# Step envs.
		promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
		obs, _, done = zip(*[p()[:3] for p in promises])
		obs = list(obs)
		done = np.stack(done)
		episode += int(done.sum())
		length += 1
		step += (done * length).sum()
		length *= (1 - done)
	# Return new state to allow resuming the simulation.
	return (step - steps, episode - episodes, done, length, obs, agent_state)