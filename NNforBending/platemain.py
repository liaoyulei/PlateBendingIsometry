import torch
from NeuralNetwork import NeuralNetwork

def loss_fn(model, size):
	x = torch.rand(size, 2).to("cpu").requires_grad_()
	u = model(x)
	du1 = torch.autograd.grad(u[:, 0], x, torch.ones(size), True, True)[0]
	du2 = torch.autograd.grad(u[:, 1], x, torch.ones(size), True, True)[0]
	du3 = torch.autograd.grad(u[:, 2], x, torch.ones(size), True, True)[0]
	ux = torch.cat([du1[:, 0].unsqueeze(1), du2[:, 0].unsqueeze(1), du3[:, 0].unsqueeze(1)], 1)
	uy = torch.cat([du1[:, 1].unsqueeze(1), du2[:, 1].unsqueeze(1), du3[:, 1].unsqueeze(1)], 1)	
	du1x = torch.autograd.grad(du1[:, 0], x, torch.ones(size), True, True)[0]
	du1y = torch.autograd.grad(du1[:, 1], x, torch.ones(size), True, True)[0]
	du2x = torch.autograd.grad(du2[:, 0], x, torch.ones(size), True, True)[0]
	du2y = torch.autograd.grad(du2[:, 1], x, torch.ones(size), True, True)[0]
	du3x = torch.autograd.grad(du3[:, 0], x, torch.ones(size), True, True)[0]
	du3y = torch.autograd.grad(du3[:, 1], x, torch.ones(size), True, True)[0]
	uxx = torch.cat([du1x[:, 0].unsqueeze(1), du2x[:, 0].unsqueeze(1), du3x[:, 0].unsqueeze(1)], 1)
	uxy = torch.cat([du1x[:, 1].unsqueeze(1), du2x[:, 1].unsqueeze(1), du3x[:, 1].unsqueeze(1)], 1)
	uyx = torch.cat([du1y[:, 0].unsqueeze(1), du2y[:, 0].unsqueeze(1), du3y[:, 0].unsqueeze(1)], 1)
	uyy = torch.cat([du1y[:, 1].unsqueeze(1), du2y[:, 1].unsqueeze(1), du3y[:, 1].unsqueeze(1)], 1)
	lossi = torch.mean(torch.sum(uxx**2 + uxy**2 + uyx**2 + uyy**2, 1)/2 - 2.5e-2*u[:, 2])
#	lossi = torch.mean((torch.sum(n*uxx, 1)**2 + 2*torch.sum(n*uxy, 1)**2 + torch.sum(n*uyy, 1)**2)/2 - 2.5e-2*u(model, x)[:, 2])
	lossc = torch.mean((torch.sum(ux**2, 1) - 1)**2 + 2*torch.sum(ux*uy, 1)**2 + (torch.sum(uy**2, 1)-1)**2)
	return lossi, lossc


model = NeuralNetwork(10, 5).to("cpu") #unit, depth
#model.load_state_dict(torch.load("model/model50000.pth"))
optimizer = torch.optim.Adam(model.parameters())
model.train()
for batch in range(50000): #iteration step
	lossi, lossc = loss_fn(model, 64)
	loss = lossi + 10 * lossc #penalty
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if (batch + 1) % 10000 == 0:
		print(batch + 1, ": ", lossi.item(), lossc.item())
		torch.save(model.state_dict(), "model/model" + str(batch+1) + ".pth")