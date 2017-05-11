#!/usr/bin/env python

from globes import projection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# All of our verices for the icosahedron
A = projection.Vertex(2, 1, 0, "A")
B = projection.Vertex(-2, 1, 0, "B")
C = projection.Vertex(2, -1, 0, "C")
D = projection.Vertex(-2, -1, 0, "D")

E = projection.Vertex(1, 0, 2, "E")
F = projection.Vertex(-1, 0, 2, "F")
G = projection.Vertex(1, 0, -2, "G")
H = projection.Vertex(-1, 0, -2, "H")

I = projection.Vertex(0, 2, 1, "I")
J = projection.Vertex(0, -2, 1, "J")
K = projection.Vertex(0, 2, -1, "K")
L = projection.Vertex(0, -2, -1, "L")

VERT = [A, B, C, D, E, F, G, H, I, J, K, L]

# All of our Faces for the icosahedron

F1 = projection.Face([H, L, D], 1)
F2 = projection.Face([H, D, B], 2)
F3 = projection.Face([H, B, K], 3)
F4 = projection.Face([H, K, G], 4)
F5 = projection.Face([H, G, L], 5)

F6 = projection.Face([C, L, G], 6)
F7 = projection.Face([G, A, C], 7)
F8 = projection.Face([A, G, K], 8)
F9 = projection.Face([K, I, A], 9)
F10 = projection.Face([I, K, B], 10)
F11 = projection.Face([B, F, I], 11)
F12 = projection.Face([F, B, D], 12)
F13 = projection.Face([D, J, F], 13)
F14 = projection.Face([J, D, L], 14)
F15 = projection.Face([L, C, J], 15)

F16 = projection.Face([E, J, C], 16)
F17 = projection.Face([E, C, A], 17)
F18 = projection.Face([E, A, I], 18)
F19 = projection.Face([E, I, F], 19)
F20 = projection.Face([E, F, J], 20)

# THE ICOSAHEDRON ITSELF
ICO = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20]
xi = np.zeros(20)
yi = np.zeros(20)
zi = np.zeros(20)

xii = np.zeros(12)
yii = np.zeros(12)
zii = np.zeros(12)

pmiddles = []


for i,f in enumerate(ICO):
    f.calcSystem()
    xi[i] = f.middle[0]
    yi[i] = f.middle[1]
    zi[i] = f.middle[2]
    #print("ID: {0} Middle Coordinates: {1}".format(f.ID, f.middle))
    pmiddles.append(projection.point_to_sphere(f.middle))
    
for j, v in enumerate(VERT):
    xii[j] = v.x
    yii[j] = v.y
    zii[j] = v.z

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xi, yi, zi, c = 'b')
#ax.scatter(xii, yii, zii, c = 'r')
#plt.show()

chosen = projection.pick_face(np.array([0.5 * np.math.pi, 0.5 * np.math.pi + 0.1]), ICO)
print("Chosen face id: {0}".format(chosen.ID))

f, s = projection.project_onto_ico([0.5 * np.math.pi, 0.5 * np.math.pi + 0.1], ICO)

