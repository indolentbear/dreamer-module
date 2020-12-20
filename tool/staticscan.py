import tensorflow as tf

def static_scan(fn, action, state, reverse=False):
	"""
	Aruguments:
		fn      :   function, img_step, 训练出的模型向后预测action序列步
							    obs_step,
		action  :
		state   :
	"""
	lastS = state
	outputs = [[] for _ in tf.nest.flatten(state)]
	indices = range(len(tf.nest.flatten(action)[0]))
	if reverse:
		indices = reversed(indices)
	for index in indices:
		inp = tf.nest.map_structure(lambda x: x[index], action)
		lastS = fn(lastS, inp)
		[o.append(l) for o, l in zip(outputs, tf.nest.flatten(lastS))]
	if reverse:
		outputs = [list(reversed(x)) for x in outputs]
	outputs = [tf.stack(x, 0) for x in outputs]
	return tf.nest.pack_sequence_as(state, outputs)
