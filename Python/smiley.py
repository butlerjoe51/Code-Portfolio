import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1, aspect=1)

ax.scatter([.5],[.5], c='#FFCC00', s=120000, label="face")
ax.scatter([.35, .65], [.63, .63], c='k', s=1000, label="eyes")

X = np.linspace(.3, .7, 100)
Y = 2* (X-.5)**2 + 0.30

ax.plot(X, Y, c='k', linewidth=8, label="smile")

ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
