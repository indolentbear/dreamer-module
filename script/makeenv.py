import tool
import wrappers

def make_env(config, writer, prefix, datadir, store):
	suite, task = config.task.split('_', 1)
	if suite == 'dmc':
		env = wrappers.DeepMindControl(task)
		env = wrappers.ActionRepeat(env, config.action_repeat)
		env = wrappers.NormalizeActions(env)
	elif suite == 'atari':
		env = wrappers.Atari(
			task, config.action_repeat, (64, 64), grayscale=False,
			life_done=True, sticky_actions=True)
		env = wrappers.OneHotAction(env)
	else:
		raise NotImplementedError(suite)
	# 限制单个episode的步数
	env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
	callbacks = []
	if store:
		callbacks.append(lambda ep: tool.save_episodes(datadir, [ep]))
	callbacks.append(
		lambda ep: tool.summarize_episode(ep, config, datadir, writer, prefix))
	# collect data
	env = wrappers.Collect(env, callbacks, config.precision)
	# reward shape
	env = wrappers.RewardObs(env)
	return env