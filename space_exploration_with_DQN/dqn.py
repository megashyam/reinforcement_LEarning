import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
	
	def __init__(self, inp_state_dims, out_action_dims, hidden_dims=128, enable_dueling_dqn=False, fc_dueling=None):
		super(DQN, self).__init__()
		self.enable_dueling_dqn=enable_dueling_dqn
		self.fc1=nn.Linear(inp_state_dims, hidden_dims)
		self.ln1=nn.LayerNorm(hidden_dims)
		# self.fc2=nn.Linear(hidden_dims, hidden_dims)
		# self.ln2=nn.LayerNorm(hidden_dims)

		if self.enable_dueling_dqn:
			self.fc_value=nn.Linear(hidden_dims, 1)
			#self.value=nn.Linear(fc_dueling, 1)

			self.fc_advantage=nn.Linear(hidden_dims, out_action_dims)
			#self.advantage=nn.Linear(fc_dueling, out_action_dims)

		self.outs=nn.Linear(hidden_dims, out_action_dims)

	def forward(self, x):
		x=self.ln1(self.fc1(x))
		x=F.relu(x)

		if self.enable_dueling_dqn:
			V=self.fc_value(x)
			#V=self.value(v)

			A=self.fc_advantage(x)
			#A=self.advantage(a)

			Q = V + A - torch.mean(A, dim=1, keepdim=True)
		else:

			Q=self.outs(x)

		return Q

if __name__ == '__main__':
	
	state_dims=8
	outs=4

	dqn=DQN(state_dims, outs)

	state=torch.randn(1, state_dims)
	print(state)

	outs=dqn(state)
	print(outs)
