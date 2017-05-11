#!/usr/bin/env python
import numpy as np


class Face(object):
    """         
    Face of a polyhedron, defined by its vertices. 
    
    Attributes
    ----------
    vertices: list
        list of Vertex objects confining the face
    ID: int
        number identifying the Face
    middle: np array
        middle of the face
        also origin of the local coordinate system
    normal: np array
        normal vector to the face
    u: np array
        first axis of the local coordinate system
    v: np array
        second axis of the local coordinate system
    """

    def __init__(self, vertices, ID):
        """
        Creates a Surface Object from a set of vertices
        
        Parameters
        ----------
        vertices: list
            list of Vertex objects confining the face
        ID: int
            number identifying the Face
        """
        
        self.vertices = vertices
        self.numVertices = len(self.vertices)
        self.ID = ID

    def calcSystem(self):
        """
        Calculates middle of the face, normal vector
        and local coordinate system
        """        
        
        # middle as average of the vertices
        self.middle = np.zeros(3)
        for v in self.vertices:
            self.middle += v.coordinates
        self.middle = self.middle / self.numVertices
        
        # normal is in the direction of the origin through the middle
        self.normal = self.middle / np.linalg.norm(self.middle)
        
        # pick first axis from middle to first vertex
        # pick second axis orthogonal to first and to normal
        self.u = self.middle - self.vertices[0].coordinates
        self.v = np.outer(self.normal, self.u)


class Vertex(object):

    def __init__(self, x, y, z, name):
        """
        Creates the Vertex Object with Carthesian coordinates
        
        Parameters
        ----------
        x: float
            x coordinate
        y: float
            y coordinate
        z: float
            z coordinate
        coordinates: np array
            (x, y, z) coordinates
        name: string
        
        Attributes
        ----------
        x: float
            x coordinate
        y: float
            y coordinate
        z: float
            z coordinate
        name: string
            single letter name
            for example, 'A'
        """
        
        self.x = x
        self.y = y
        self.z = z
        self.coordinates = np.array([self.x, self.y, self.z])
        self.name = name

    def to_spherical(self):
        """
        Maybe we'll need it, who knows?
        :return:
        """
        pass


def point_to_sphere(p):
    """
    Projects an input coordinates to the spherical surface.

    Arguments
    ---------
    p: Numpy Array
        x, y, z

    Returns
    -------
    Numpy Array
        theta, phi
    """

    r = np.sqrt(np.dot(p, p))
    if r == 0:
        print("Error: r should'nt be 0")
        return None

    if p[0] == 0:
        phi = np.pi/2
    else:
        phi = np.arctan(p[1]/p[0])
    theta = np.arccos(p[2]/r)

    return np.array([theta, phi])


def pick_face(p, ICO):
    """
    Picks the face of the icosahedron to project the point onto

    Arguments
    ---------
    p: Numpy Array
        theta, phi

    ICO: List
        List of Faces

    Returns
    -------
    Face
        The face that the point should be projected to.
    """

    lowest = np.pi*2
    closest_face = ICO[0]

    for f in ICO:
        pmiddle = point_to_sphere(f.middle)
        dist = np.sqrt((pmiddle[0] - p[0])**2. + (pmiddle[1] - p[1])**2.)
        if dist < lowest:
            lowest = dist
            closest_face = f

    return closest_face

