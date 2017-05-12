#!/usr/bin/env python

from globes import projection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

golden = (1. + np.sqrt(5)) / 2.0

# All of our verices for the icosahedron
A = projection.Vertex(golden, 1, 0, "A")
B = projection.Vertex(-golden, 1, 0, "B")
C = projection.Vertex(golden, -1, 0, "C")
D = projection.Vertex(-golden, -1, 0, "D")

E = projection.Vertex(1, 0, golden, "E")
F = projection.Vertex(-1, 0, golden, "F")
G = projection.Vertex(1, 0, -golden, "G")
H = projection.Vertex(-1, 0, -golden, "H")

I = projection.Vertex(0, golden, 1, "I")
J = projection.Vertex(0, -golden, 1, "J")
K = projection.Vertex(0, golden, -1, "K")
L = projection.Vertex(0, -golden, -1, "L")

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

def middles_test(ICO):
    """
    Test to check whether projected middles of faces get projected back
    onto the original middles
    
    Arguments
    ---------
    ICO: list
        list of Face objects that form an icosahedron
    """
    middles = []
    proj_middles = []
    pback_middles = []


    for i,f in enumerate(ICO):
        f.calcSystem()
        middles.append(f.middle)
    
        pmiddle = projection.point_to_sphere(f.middle)
        proj_middles.append(pmiddle)
        
        pback = projection.project_to_face(pmiddle, f)
        pback_middles.append(pback)

    test = np.allclose(middles, pback_middles, rtol=1.e-12)
    if test:
        print "Test succesfull, middles projected back onto middles"
    else:
        print "Test failed, middles projected back elsewhere than \
        original middles"

# calculate local coordinate system etc
for f in ICO:
    f.calcSystem()
    
def plot_lcs(ICO):    
    """
    Plot the vertices and middles of each face in their 
    local coordinate systems
    
    Arguments
    ---------
    ICO: list
        list of Face objects that form an icosahedron
    """
    colors = ['b', 'r', 'g', 'k']
    fig, axarr = plt.subplots(nrows=4, ncols=5)
    
    for j, f in enumerate(ICO):
        verts = f.calcLocalVertices()
        for i, v in enumerate(verts):
            axarr[j/5][j%5].scatter(v[0], v[1], c = colors[i])
        axarr[j/5][j%5].scatter(0., 0., c = colors[-1])
    fig.show()
    



## test part to see what a projected circle looks like
#num = 1000
#circle_phi = np.array([0.661 for n in range(num)])
#circle_theta = np.linspace(0.0, 2 * np.math.pi, num=num)
#
#face_points = [[] for f in ICO]
#all_points = []
#for n in range(num):
#    p = np.array([circle_phi[n], circle_theta[n]])
#    f, projp = projection.project_onto_ico(p, ICO)
#    face_points[f.ID - 1].append(projp)
#    all_points.append(projp)
#
#for i in range(20):
#    print("{0} - {1}".format(i+1,len(face_points[i])))
##
## print(len(face_points[18]))
## print(len(face_points[19]))
##
## face19_points = np.array(face_points[18])
## face19 = ICO[18]
##
## face18_points = np.array(face_points[17])
#face18 = ICO[17]
#
#import visualize
#
#all_points = np.array(all_points)
#visualize.draw_face(face_points, ICO)