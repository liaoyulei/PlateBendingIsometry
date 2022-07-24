import torch
from torch import nn
class NeuralNetwork(nn.Module):
	def __init__(self, unit, depth):
		super(NeuralNetwork, self).__init__()
		self.depth = depth
		self.linear1 = nn.Linear(2, unit)
		self.linears = nn.ModuleList([nn.Linear(unit, unit) for i in range(2*depth)])
		self.linear2 = nn.Linear(unit, 3)

	def forward(self, x):
		u = torch.tanh(self.linear1(x))
		for i in range(self.depth):
			uu = torch.tanh(self.linears[2*i](u))
			u = u + torch.tanh(self.linears[2*i+1](uu))
		return torch.prod(x**2, 1).unsqueeze(1) * self.linear2(u) + torch.cat([x, torch.zeros(x.shape[0], 1)], 1)