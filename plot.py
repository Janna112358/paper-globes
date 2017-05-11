from numpy import random
import matplotlib.pyplot as plt

size = 5

xs = random.random(2)
ys = random.random(2)
mags = [-1.44, 10.71]
mags = [-mag+12.15 for mag in mags]

plt.scatter(xs, ys, s=mags, c='k')
plt.axis('off')
plt.savefig('paper_globe.pdf', bbox_inches='tight')