#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np


def draw_face(points, face):
    """
    Draw a matplotlib graph of a single face with points on it, also draw the vertices and Patch the edges of the
    triangle

    Arguments
    ---------
    points: NumPy Array
        The points to draw

    face: Face
        The face object

    Returns
    -------
    Void
    """


    # the line below might not work, but we need to add [0.0, 0.0] to the very end of the numpy matrix
    vertices = face.calcLocalVertices()
    new_vertices = []
    for v in vertices:
        new_vertices.append(np.array([-v[1], -v[0]]))
    new_vertices.append(np.array([0.0, 0.0]))



    # this is the mask to draw the triangle. the last line actually closes the poly
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
         ]
    path = Path(new_vertices, codes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='white', lw=1, alpha=0.5)
    ax.add_patch(patch)
    ax.scatter(-points[:,1], -points[:,0])

    plt.show()