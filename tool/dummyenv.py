import gym
import numpy as np

class DummyEnv:

	def __init__(self):
		self._random = np.random.RandomState(seed=0)
		self._step = None

	@property
	def observation_space(self):
		low = np.zeros([64, 64, 3], dtype=np.uint8)
		high = 255 * np.ones([64, 64, 3], dtype=np.uint8)
		spaces = {'image': gym.spaces.Box(low, high)}
		return gym.spaces.Dict(spaces)

	@property
	def action_space(self):
		low = -np.ones([5], dtype=np.float32)
		high = np.ones([5], dtype=np.float32)
		return gym.spaces.Box(low, high)

	def reset(self):
		self._step = 0
		obs = self.observation_space.sample()
		return obs

	def step(self, action):
		obs = self.observation_space.sample()
		reward = self._random.uniform(0, 1)
		self._step += 1
		done = self._step >= 1000
		info = {}
		return obs, reward, done, info
