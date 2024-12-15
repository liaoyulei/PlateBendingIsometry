import torch
import numpy as np
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_fn(model, alpha, size, num_subareas, ith_area):
	x, num_pts = generate_randomPointsInO(model, size, num_subareas, ith_area)
	x = x.requires_grad_()
	u = model(x)
	du1 = torch.autograd.grad(u[:, 0], x, torch.ones(num_pts, device = device), True, True)[0]
	du2 = torch.autograd.grad(u[:, 1], x, torch.ones(num_pts, device = device), True, True)[0]
	du3 = torch.autograd.grad(u[:, 2], x, torch.ones(num_pts, device = device), True, True)[0]
	ux = torch.cat([du1[:, 0].unsqueeze(1), du2[:, 0].unsqueeze(1), du3[:, 0].unsqueeze(1)], 1)
	uy = torch.cat([du1[:, 1].unsqueeze(1), du2[:, 1].unsqueeze(1), du3[:, 1].unsqueeze(1)], 1)
	n = torch.cross(ux, uy)
	du1x = torch.autograd.grad(du1[:, 0], x, torch.ones(num_pts, device = device), True, True)[0]
	du1y = torch.autograd.grad(du1[:, 1], x, torch.ones(num_pts, device = device), True, True)[0]
	du2x = torch.autograd.grad(du2[:, 0], x, torch.ones(num_pts, device = device), True, True)[0]
	du2y = torch.autograd.grad(du2[:, 1], x, torch.ones(num_pts, device = device), True, True)[0]
	du3x = torch.autograd.grad(du3[:, 0], x, torch.ones(num_pts, device = device), True, True)[0]
	du3y = torch.autograd.grad(du3[:, 1], x, torch.ones(num_pts, device = device), True, True)[0]
	uxx = torch.cat([du1x[:, 0].unsqueeze(1), du2x[:, 0].unsqueeze(1), du3x[:, 0].unsqueeze(1)], 1)
	uxy = torch.cat([du1x[:, 1].unsqueeze(1), du2x[:, 1].unsqueeze(1), du3x[:, 1].unsqueeze(1)], 1)
	uyx = torch.cat([du1y[:, 0].unsqueeze(1), du2y[:, 0].unsqueeze(1), du3y[:, 0].unsqueeze(1)], 1)
	uyy = torch.cat([du1y[:, 1].unsqueeze(1), du2y[:, 1].unsqueeze(1), du3y[:, 1].unsqueeze(1)], 1)
	lossi = torch.mean((torch.sum(n*uxx, 1) - alpha)**2 + 2*torch.sum(n*uxy, 1)**2 + (torch.sum(n*uyy, 1) - alpha)**2)/2
	lossc = torch.mean((torch.sum(ux**2, 1) - 1)**2 + 2*torch.sum(ux*uy, 1)**2 + (torch.sum(uy**2, 1)-1)**2)
	area1 = 2 * model.width * (model.length - model.inner_length)
	area2 = 2 * model.inner_length * (model.width - model.inner_width)
	if (ith_area == 1):
		area = area1
	elif (ith_area > 1 and ith_area < num_subareas):
		area = area1 + 2 * area2 * (ith_area - 1)/(num_subareas -2)
	else:
		area = 2 *(area1 + area2)
	return area * lossi, area * lossc
	
def generate_randomPointsInO(model, size, num_subareas, ith_area):
	num1 = int(np.ceil((model.length - model.inner_length) * 2 * model.width * size))
	numMid = int(np.ceil(2 * model.inner_length * (model.width - model.inner_width) * size))
	x1 = torch.cat([(model.length - model.inner_length) * torch.rand(num1, 1).to(device) - model.length, 2 * model.width * torch.rand(num1, 1).to(device) - model.width], 1)
	xMit = torch.cat([2 * model.inner_length * (ith_area - 1) / (num_subareas - 2) * torch.rand(int(numMid * (ith_area - 1)/(num_subareas-2)), 1).to(device) - model.inner_length, (model.width - model.inner_width) * torch.rand(int(numMid * (ith_area - 1)/(num_subareas-2)), 1).to(device) + model.inner_width], 1)
	xMib = torch.cat([2 * model.inner_length * (ith_area - 1) / (num_subareas - 2) * torch.rand(int(numMid * (ith_area - 1)/(num_subareas-2)), 1).to(device) - model.inner_length, (model.width - model.inner_width) * torch.rand(int(numMid * (ith_area - 1)/(num_subareas-2)), 1).to(device) - model.width], 1)
	xMt = torch.cat([2 * model.inner_length  * torch.rand(numMid, 1).to(device) - model.inner_length, (model.width - model.inner_width) * torch.rand(numMid, 1).to(device) + model.inner_width], 1)
	xMb = torch.cat([2 * model.inner_length  * torch.rand(numMid, 1).to(device) - model.inner_length, (model.width - model.inner_width) * torch.rand(numMid, 1).to(device) - model.width], 1)
	x2 = torch.cat([(model.length - model.inner_length) * torch.rand(num1, 1).to(device) + model.inner_length, 2 * model.width * torch.rand(num1, 1).to(device) - model.width], 1)
	if (ith_area == 1):
		x = x1
		num_pts = num1
	elif (ith_area < num_subareas and ith_area > 1):
		x = torch.cat([x1, xMit, xMib], 0)
		num_pts = num1 + 2 * int(numMid * (ith_area - 1)/(num_subareas-2))
	else:
		x = torch.cat([x1, xMt, xMb, x2], 0)
		num_pts = 2 * (num1 + numMid)
	return x, num_pts


length = 5
inner_length = 4
width = 2
inner_width = 1
size = 16
beta = 500
alpha = 5
num_subareas = 5

model = NeuralNetwork(10, 5, length, width, inner_length, inner_width, 3).to(device)
model.load_state_dict(torch.load("model/model5-thpre50000.pth"))
optimizer = torch.optim.Adam(model.parameters())
model.train()
'''
for k in range(num_subareas):
	for batch in range(50000): 
		lossi, lossc = loss_fn(model, alpha, 16, num_subareas, k + 1)
		loss = lossi + beta * lossc #penalty
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (batch + 1) % 10000 == 0:
			print(batch + 1, ": ", lossi.item(), lossc.item())
			torch.save(model.state_dict(), "model/model" + str(k + 1) +"-thpre" + str(batch+1) +  ".pth")
'''
for batch in range(1000000):
	#iteration step
	lossi, lossc = loss_fn(model, alpha, 16, num_subareas, num_subareas)
	loss = lossi + beta * lossc #penalty
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if (batch + 1) % 10000 == 0:
		print(batch + 1, ": ", lossi.item(), lossc.item())
		torch.save(model.state_dict(), "model/model" + str(batch+1) +  ".pth")

model.eval()
lossi, lossc = loss_fn(model, alpha, 10000, num_subareas, num_subareas)
print(lossi.item(), lossc.item())

		