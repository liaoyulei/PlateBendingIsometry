import torch
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
#unit, depth, length, width, inner_length, inner_width, out
ax = plt.figure().add_subplot(projection="3d")
model = NeuralNetwork(10, 5, 5, 2, 4, 1, 3)
modelbd = NeuralNetwork(10, 0, 5, 2, 4, 1, 1)
modelbd.load_state_dict(torch.load("model/modelbd.pth", map_location=torch.device('cpu')))
m = [101, 101, 200*(model.length - 1)+1, 200*(model.length - 1)+1]
n = [200*model.width+1, 200*model.width+1, 101, 101]

x = np.linspace(-model.length, 1-model.length, m[0]).astype(np.float32)
y = np.linspace(-model.width, model.width, n[0]).astype(np.float32)
x, y = np.meshgrid(x, y)
xx0 = torch.from_numpy(np.array([x.ravel(), y.ravel()]).T)

x = np.linspace(model.length-1, model.length, m[1]).astype(np.float32)
y = np.linspace(-model.width, model.width, n[1]).astype(np.float32)
x, y = np.meshgrid(x, y)
xx1 = torch.from_numpy(np.array([x.ravel(), y.ravel()])).T

x = np.linspace(1-model.length, model.length-1, m[2]).astype(np.float32)
y = np.linspace(-model.width, 1-model.width, n[2]).astype(np.float32)
x, y = np.meshgrid(x, y)
xx2 = torch.from_numpy(np.array([x.ravel(), y.ravel()])).T

x = np.linspace(1-model.length, model.length-1, m[3]).astype(np.float32)
y = np.linspace(model.width-1, model.width, n[3]).astype(np.float32)
x, y = np.meshgrid(x, y)
xx3 = torch.from_numpy(np.array([x.ravel(), y.ravel()])).T

xx = [xx0, xx1, xx2, xx3]

modelbd.eval()
model.eval()
for i in range(1, 101):
	step = str(10000 * i)
	model.load_state_dict(torch.load("model/model" + step +".pth", map_location=torch.device('cpu')))
	plt.cla()
	for j in range(4):
		with torch.no_grad():
			u = modelbd(xx[j]) * model(xx[j])
			u[:, : 2] = u[:, : 2] + xx[j]
			x, y, z = u.T
		x = x.numpy().reshape(n[j], m[j])
		y = y.numpy().reshape(n[j], m[j])
		z = z.numpy().reshape(n[j], m[j])
		ax.plot_surface(x, y, z)
	ax.set_title("step: " + step)
	plt.pause(1)
plt.show()