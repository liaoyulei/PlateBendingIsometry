import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import NeuralNetwork


model = NeuralNetwork(10, 5)
model.load_state_dict(torch.load("model/model50000.pth"))
m = 101
n = 101
x = np.linspace(0, 1, m).astype(np.float32)
y = np.linspace(0, 1, n).astype(np.float32)
x, y = np.meshgrid(x, y)
model.eval()
with torch.no_grad():
	z = model(torch.from_numpy(np.array([x.ravel(), y.ravel()]).T))
z = z.numpy().reshape(n, m)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure = False)
fig.add_axes(ax)
ax.plot_surface(x, y, z)
plt.show()