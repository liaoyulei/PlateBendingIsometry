'''
Solve the Possion equation u=sin(pi x)sin(pi y)
-Delta u=f in Omega=(0,1)^2
u=0 on partialOmega
'''
import torch
from NeuralNetwork import NeuralNetwork

def loss_fn(model, size):
	xi = torch.rand(size, 2).requires_grad_()
	y = model(xi)
	dy = torch.autograd.grad(y, xi, torch.ones(size), True, True)[0]
	f = 2 * torch.pi ** 2 * torch.prod(torch.sin(torch.pi*xi), 1)
	lossi = torch.mean(torch.sum(dy ** 2, 1) / 2 -  f * y)
	xb = torch.rand(4*size, 2)
	xb[: size, 0] = 0
	xb[size: 2*size, 0] = 1
	xb[2*size: 3*size, 1] = 0
	xb[3*size: , 1] = 1
	lossb = 4 * torch.mean(model(xb) ** 2)
	return lossi + 500 * lossb #penalty

model = NeuralNetwork(10, 5).to("cpu") #unit, depth
optimizer = torch.optim.Adam(model.parameters())
model.train()
for batch in range(50000): #iteration step
	loss = loss_fn(model, 64)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if (batch + 1) % 10000 == 0:
		print(batch + 1, ": ", loss.item())
		torch.save(model.state_dict(), "model/model" + str(batch+1) + ".pth")