import tensorflow as tf

class SampleDist:

	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return 'SampleDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		samples = self._dist.sample(self._samples)
		return tf.reduce_mean(samples, 0)

	def mode(self):
		sample = self._dist.sample(self._samples)
		logprob = self._dist.log_prob(sample)
		return tf.gather(sample, tf.argmax(logprob))[0]

	def entropy(self):
		sample = self._dist.sample(self._samples)
		logprob = self.log_prob(sample)
		return -tf.reduce_mean(logprob, 0)
