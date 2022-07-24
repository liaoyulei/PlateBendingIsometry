import torch
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

ax = plt.figure().add_subplot(projection="3d")
model = NeuralNetwork(10, 5, 5, 2)
modelbd = NeuralNetwork(10, 0, 5, 2)
m = 200 * model.length + 1
n = 200 * model.width + 1
x = np.linspace(-model.length, model.length, m).astype(np.float32)
y = np.linspace(-model.width, model.width, n).astype(np.float32)
x, y = np.meshgrid(x, y)
xx = torch.from_numpy(np.array([x.ravel(), y.ravel()]).T)
model.eval()
for i in range(100):
	step = str(10000 * (i + 1))
	model.load_state_dict(torch.load("model/model" + step +".pth"))
	modelbd.load_state_dict(torch.load("model/modelbd" + step +".pth"))
	with torch.no_grad():
		x, y, z = (xx[:, 0] + 5)**2 * model(xx).T + modelbd(xx).T
	x = x.numpy().reshape(n, m)
	y = y.numpy().reshape(n, m)
	z = z.numpy().reshape(n, m)
	plt.cla()
	ax.plot_surface(x, y, z)
	ax.set_title("step: " + step)
	plt.pause(1)
plt.show()