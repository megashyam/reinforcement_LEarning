from collections import deque
import random

class BufferMemory:
	def __init__(self, memory_size=10000):
		self.buffer = deque(maxlen=memory_size)

	def __len__(self):
		return len(self.buffer)

	def append(self, x):
		self.buffer.append(x)

	def sample(self, sample_size):
		return random.sample(self.buffer, sample_size)
		
