import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

class OneHotDist:

	def __init__(self, logits=None, probs=None):
		self._dist = tfd.Categorical(logits=logits, probs=probs)
		self._num_classes = self.mean().shape[-1]
		self._dtype = prec.global_policy().compute_dtype

	@property
	def name(self):
		return 'OneHotDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def prob(self, events):
		indices = tf.argmax(events, axis=-1)
		return self._dist.prob(indices)

	def log_prob(self, events):
		indices = tf.argmax(events, axis=-1)
		return self._dist.log_prob(indices)

	def mean(self):
		return self._dist.probs_parameter()

	def mode(self):
		return self._one_hot(self._dist.mode())

	def sample(self, amount=None):
		amount = [amount] if amount else []
		indices = self._dist.sample(*amount)
		sample = self._one_hot(indices)
		probs = self._dist.probs_parameter()
		sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
		return sample

	def _one_hot(self, indices):
		return tf.one_hot(indices, self._num_classes, dtype=self._dtype)