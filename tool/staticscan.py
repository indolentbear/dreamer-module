import tensorflow as tf

def static_scan(fn, inputs, start, reverse=False):
	last = start
	outputs = [[] for _ in tf.nest.flatten(start)]
	indices = range(len(tf.nest.flatten(inputs)[0]))
	if reverse:
		indices = reversed(indices)
	for index in indices:
		inp = tf.nest.map_structure(lambda x: x[index], inputs)
		last = fn(last, inp)
		[o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
	if reverse:
		outputs = [list(reversed(x)) for x in outputs]
	outputs = [tf.stack(x, 0) for x in outputs]
	return tf.nest.pack_sequence_as(start, outputs)
