import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

def preprocess(obs, config):
	dtype = prec.global_policy().compute_dtype
	obs = obs.copy()
	with tf.device('cpu:0'):
		obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
		clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
		obs['reward'] = clip_rewards(obs['reward'])
	return obs