import torch
from NeuralNetwork import NeuralNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_fn(model, modelbd, alpha, size):
	x = torch.rand(model.square * size, 2)
	for i in range(2 * model.length):
		for j in range(2 * model.width):
			x[(2*i*model.width + j) * size: (2*i*model.width + j + 1) * size, :] += torch.tensor([i - model.length, j - model.width])
	x = x.to(device).requires_grad_()
	u = (x[:, 0].unsqueeze(1) + model.length)**2 * model(x) + modelbd(x)
	du1 = torch.autograd.grad(u[:, 0], x, torch.ones(model.square * size), True, True)[0]
	du2 = torch.autograd.grad(u[:, 1], x, torch.ones(model.square * size), True, True)[0]
	du3 = torch.autograd.grad(u[:, 2], x, torch.ones(model.square * size), True, True)[0]
	ux = torch.cat([du1[:, 0].unsqueeze(1), du2[:, 0].unsqueeze(1), du3[:, 0].unsqueeze(1)], 1)
	uy = torch.cat([du1[:, 1].unsqueeze(1), du2[:, 1].unsqueeze(1), du3[:, 1].unsqueeze(1)], 1)
	n = torch.cross(ux, uy)
	du1x = torch.autograd.grad(du1[:, 0], x, torch.ones(model.square * size), True, True)[0]
	du1y = torch.autograd.grad(du1[:, 1], x, torch.ones(model.square * size), True, True)[0]
	du2x = torch.autograd.grad(du2[:, 0], x, torch.ones(model.square * size), True, True)[0]
	du2y = torch.autograd.grad(du2[:, 1], x, torch.ones(model.square * size), True, True)[0]
	du3x = torch.autograd.grad(du3[:, 0], x, torch.ones(model.square * size), True, True)[0]
	du3y = torch.autograd.grad(du3[:, 1], x, torch.ones(model.square * size), True, True)[0]
	uxx = torch.cat([du1x[:, 0].unsqueeze(1), du2x[:, 0].unsqueeze(1), du3x[:, 0].unsqueeze(1)], 1)
	uxy = torch.cat([du1x[:, 1].unsqueeze(1), du2x[:, 1].unsqueeze(1), du3x[:, 1].unsqueeze(1)], 1)
	uyx = torch.cat([du1y[:, 0].unsqueeze(1), du2y[:, 0].unsqueeze(1), du3y[:, 0].unsqueeze(1)], 1)
	uyy = torch.cat([du1y[:, 1].unsqueeze(1), du2y[:, 1].unsqueeze(1), du3y[:, 1].unsqueeze(1)], 1)
	lossi = torch.mean((torch.sum(n*uxx, 1) - alpha)**2 + 2*torch.sum(n*uxy, 1)**2 + (torch.sum(n*uyy, 1) - alpha)**2) / 2
#	lossi = torch.mean(torch.sum(uxx**2 + uxy**2 + uyx**2 + uyy*2, 1)/2 - alpha*torch.sum((uxx + uyy)*n, 1)) + alpha**2
	lossc = torch.mean((torch.sum(ux**2, 1) - 1)**2 + 2*torch.sum(ux*uy, 1)**2 + (torch.sum(uy**2, 1) - 1)**2)
	return model.square * lossi, model.square * lossc

def loss_bd(model, size):
	x = torch.rand(2 * model.width * size, 2)
	x[:, 0] = -model.length
	for j in range(2 * model.width):
		x[j * size: (j + 1) * size, : 1] += j - model.width
	x = x.to(device).requires_grad_()
	u = model(x)
	du1 = torch.autograd.grad(u[:, 0], x, torch.ones(2 * model.width * size), True, True)[0]
	du2 = torch.autograd.grad(u[:, 1], x, torch.ones(2 * model.width * size), True, True)[0]
	du3 = torch.autograd.grad(u[:, 2], x, torch.ones(2 * model.width * size), True, True)[0]	
	return 2*model.width * torch.mean(torch.sum((u[:, : 2] - x)**2 + du3**2, 1) + u[:, 2]**2 + (du1[:, 0] - 1)**2 + du1[:, 1]**2 + du2[:, 0]**2 + (du2[:, 1] - 1)**2)

model = NeuralNetwork(10, 5, 5, 2).to(device) #unit, depth, length, width
modelbd = NeuralNetwork(10, 0, 5, 2).to(device)
#model.load_state_dict(torch.load("model/model900000.pth")) #you can load a model and continue training it, without retraining again
#modelbd.load_state_dict(torch.load("model/modelbd900000.pth"))
optimizer = torch.optim.Adam(model.parameters())
optimizerbd = torch.optim.Adam(modelbd.parameters())
model.train()
modelbd.train()
for batch in range(1000000): #iteration step
	for k in range(10): # the inner iteration step may be smaller
		lossb = loss_bd(modelbd, 16)
		optimizerbd.zero_grad()
		lossb.backward()
		optimizerbd.step()
	lossi, lossc = loss_fn(model, modelbd, 2.5, 16)
	lossb = loss_bd(modelbd, 16)
	loss = lossi + 1000 * (lossc + lossb) #penalty
	optimizer.zero_grad()
	optimizerbd.zero_grad()
	loss.backward()
	optimizer.step()
	optimizerbd.step()
	if (batch + 1) % 10000 == 0:
		print(batch + 1, ": ", lossi.item(), lossc.item(), lossb.item())
		torch.save(model.state_dict(), "model/model" + str(batch+1) + ".pth")	
		torch.save(modelbd.state_dict(), "model/modelbd" + str(batch+1) + ".pth")		