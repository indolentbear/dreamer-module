import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

class Adam(tf.Module):

	def __init__(self, name, modules, lr, clip=None, wd=None, wdpattern=r'.*'):
		self._name = name
		self._modules = modules
		self._clip = clip
		self._wd = wd
		self._wdpattern = wdpattern
		self._opt = tf.optimizers.Adam(lr)
		self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')
		self._variables = None

	@property
	def variables(self):
		return self._opt.variables()

	def __call__(self, tape, loss):
		if self._variables is None:
			variables = [module.variables for module in self._modules]
			self._variables = tf.nest.flatten(variables)
			count = sum(np.prod(x.shape) for x in self._variables)
			print(f'Found {count} {self._name} parameters.')
		assert len(loss.shape) == 0, loss.shape
		with tape:
			loss = self._opt.get_scaled_loss(loss)
		grads = tape.gradient(loss, self._variables)
		grads = self._opt.get_unscaled_gradients(grads)
		norm = tf.linalg.global_norm(grads)
		if self._clip:
			grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
		if self._wd:
			context = tf.distribute.get_replica_context()
			context.merge_call(self._apply_weight_decay)
		self._opt.apply_gradients(zip(grads, self._variables))
		return norm

	def _apply_weight_decay(self, strategy):
		print('Applied weight decay to variables:')
		for var in self._variables:
			if re.search(self._wdpattern, self._name + '/' + var.name):
				print('- ' + self._name + '/' + var.name)
				strategy.extended.update(var, lambda var: self._wd * var)
