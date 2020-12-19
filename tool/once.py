class Once:

	def __init__(self):
		self._once = True

	def __call__(self):
		if self._once:
			self._once = False
			return True
		return False