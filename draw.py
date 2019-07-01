# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

output = torch.load("data/out.pt").numpy()
target = torch.load("data/targ.pt").numpy()
trained_points = torch.load("data/trained_points.pt").numpy()
pts1 = output[0,:,:].reshape(1, output.shape[1], output.shape[2])
pts2 = target[0,:,:].reshape(1, target.shape[1], target.shape[2])
pts3 = trained_points

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts1[0,:,0], pts1[0,:,1], pts1[0,:,2], marker='o')
ax.set_title('output')
plt.draw()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts2[0,:,0], pts2[0,:,1], pts2[0,:,2], marker='o')
ax.set_title('target')
plt.draw()

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3[0,:,0], pts3[0,:,1], pts3[0,:,2], marker='o')
ax.set_title('trained')
plt.draw()

plt.show()
#
#
for i in range(5):
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target[i,:,0], target[i,:,1], target[i,:,2], marker='o')
plt.show()
