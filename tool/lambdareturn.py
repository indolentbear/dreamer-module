import tensorflow as tf
import tool

def lambda_return(
		reward, value, pcont, bootstrap, lambda_, axis):
	"""

	:param reward		:
	:param value		: V值
	:param pcont		: 折扣
	:param bootstrap	:
	:param lambda_	: 单步/MC多步
	:param axis		: 横/纵
	:return returns	: 累计汇报
	"""
	# lambda_ = 1, Monte Carlo return.
	# lambda_ = 0, fixed 1-step return
	assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
	if isinstance(pcont, (int, float)):
		pcont = pcont * tf.ones_like(reward)
	dims = list(range(reward.shape.ndims))
	dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
	if axis != 0:
		reward = tf.transpose(reward, dims)
		value = tf.transpose(value, dims)
		pcont = tf.transpose(pcont, dims)
	if bootstrap is None:
		bootstrap = tf.zeros_like(value[-1])
	next_values = tf.concat([value[1:], bootstrap[None]], 0)
	inputs = reward + pcont * next_values * (1 - lambda_)
	returns = tool.static_scan(
		lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
		(inputs, pcont), bootstrap, reverse=True)
	if axis != 0:
		returns = tf.transpose(returns, dims)
	return returns
