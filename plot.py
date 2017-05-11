
from numpy import random
import math
import matplotlib.pyplot as plt

scale = 1

# Create net
def node(x,y):
    return [x*scale,y*math.sqrt(3)*scale]


def line(k1, k2):
    x = [k1[0],k2[0]]
    y = [k1[1],k2[1]]
    plt.plot(x, y, 'k-')


l1 = node(0,2)
l2 = node(10,2)
j1 = node(1,1)
j2 = node(11,1)
h1 = node(1,3)
h2 = node(3,3)
h3 = node(5,3)
h4 = node(7,3)
h5 = node(9,3)
e1 = node(2,0)
e2 = node(4,0)
e3 = node(6,0)
e4 = node(8,0)
e5 = node(10,0)

line(l1,l2)
line(j1,j2)
line(l1,e1)
line(h1,e2)
line(h2,e3)
line(h3,e4)
line(h4,e5)
line(h5,j2)
line(l1,h1)
line(j1,h2)
line(e1,h3)
line(e2,h4)
line(e3,h5)
line(e4,l2)
line(e5,j2)

plt.axis((l1[0],j2[0],e1[1],h1[1]))

# size = 5
# xs = random.random(2)
# ys = random.random(2)
# mags = [-1.44, 10.71]
# mags = [(-mag+12.15)*5 for mag in mags]
#plt.scatter(xs, ys, s=mags, c='k')
plt.axis('off')
plt.savefig('paper_globe.pdf', bbox_inches='tight')