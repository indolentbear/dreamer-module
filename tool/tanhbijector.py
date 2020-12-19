import tensorflow as tf
import tensorflow_probability as tfp

class TanhBijector(tfp.bijectors.Bijector):

	def __init__(self, validate_args=False, name='tanh'):
		super().__init__(
			forward_min_event_ndims=0,
			validate_args=validate_args,
			name=name)

	def _forward(self, x):
		return tf.nn.tanh(x)

	def _inverse(self, y):
		dtype = y.dtype
		y = tf.cast(y, tf.float32)
		y = tf.where(
			tf.less_equal(tf.abs(y), 1.),
			tf.clip_by_value(y, -0.99999997, 0.99999997), y)
		y = tf.atanh(y)
		y = tf.cast(y, dtype)
		return y

	def _forward_log_det_jacobian(self, x):
		log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
		return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))