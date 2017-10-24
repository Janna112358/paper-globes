#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from globes import projection


def middles_test(ICO):
    """
    Test to check whether projected middles of faces get projected back
    onto the original middles
    
    Arguments
    ---------
    ICO: Icosahedron
    """
    middles = np.zeros((20, 3))
    proj_middles = np.zeros((20, 2))
    pback_middles = np.zeros((20, 3))


    for i,f in enumerate(ICO.faces):
        middles[i] = f.middle
    
        pmiddle = projection.point_to_sphere(f.middle)
        proj_middles[i] = pmiddle
        
        face, pback = ICO.project_in_3D(pmiddle)
        if face != f:
            print("Point not projected back onto the same face")
        pback_middles[i] = pback

    test = np.allclose(middles, pback_middles, rtol=1.e-12)
    if test:
        print "Test succesfull, middles projected back onto middles"
    else:
        print "Test failed, middles projected back elsewhere than \
        original middles"

    
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
            axarr[j/5][j%5].scatter(v[0], v[1], c=colors[i])
        axarr[j/5][j%5].scatter(0., 0., c=colors[-1])
    fig.show()
    
    
if __name__ == "__main__":
    # buil an icosahedron
    ICO = projection.Icosahedron()
    
    middles_test(ICO)
    
    ## test part to see what a projected circle looks like
    num = 100
    circle_phi = np.array([0.661 for n in range(num)])
    circle_theta = np.linspace(0.0, 2 * np.math.pi, num=num)
    
    colours = ['red', 'green', 'blue', 'black', 'magenta', 'cyan', 'gray', 
               'olive', 'orange', 'purple', 'darkblue']
    
    face_points = [[] for i in range(20)]
    for n in range(num):
        p = np.array([circle_phi[n], circle_theta[n]])
        face, projp = ICO.project_in_3D(p)
        face_points[face.ID - 1].append(projp)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(20):
        for p in face_points[i]:
            ax.scatter(p[0], p[1], p[2], c = colours[i%len(colours)])

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig('test.png')
    fig.show()

