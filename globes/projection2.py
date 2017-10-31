#!/usr/bin/env python
import numpy as np
import sys
from warnings import warn


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
    size: float
        length of middle to first vertex
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
        
        # check that at least three vertices are given
        try:
            assert(self.numVertices >= 3)
        except AssertionError:
            print("Can not creatre face with less than three vertices.")
            sys.exit(1)
        
        self.calc_system()

    def calc_system(self):
        """
        Calculates middle of the face, normal vector
        and local coordinate system
        """        
        
        # middle as average of the vertices
        self.middle = np.zeros(self.numVertices)
        for v in self.vertices:
            self.middle += v.coordinates
        self.middle = self.middle / self.numVertices
        
        # normal is in the direction of the origin through the middle
        self.normal = self.middle / np.linalg.norm(self.middle)
        
        # pick frist axis from v1 to v2, second from v1 to v3
        self.u = self.vertices[1].coordinates - self.vertices[0].coordinates
        self.v = self.vertices[2].coordinates - self.vertices[0].coordinates
        
        self.angle = np.arccos(np.dot(self.u, self.v) / 
                             (np.linalg.norm(self.u) * np.linalg.norm(self.v)))
        
    def calc_local_vertices(self):
        """
        Get position of the vertices of the face in the local coordinate system
        
        Only works for (triangular) icosahedron surfaces)
        hard coded maths
        
        Returns
        -------
        list
            list of 2D arrays of the coordinates of the vertices in the local
            coodinate system
        """
        
        D = np.linalg.norm(self.middle - self.vertices[0].coordinates)
        v1 = np.array([D, 0])
        v2 = np.array([-D*np.cos(self.angle), -D*np.sin(self.angle)])
        v3 = np.array([-D*np.cos(self.angle), D*np.sin(self.angle)])
        
        return [v1, v2, v3]         
    
    def global_to_lcs(self, point):
        """
        Convert a point on the face in global (3D) coordinates to the local
        coordinate system of the face (2D)
        
        Paramters
        ---------
        point: array-like
            3D global coordinates of a point on the face
        
        Returns
        -------
        NumPy Array: the point in the lcs of the face
        """        
        
        eps = 1.0e-6
        # check that point is in the plane of the face
        assert(np.dot((point - self.middle), self.normal) < eps)
        
        start_vector = self.vertices[0].coordinates
        p = point - start_vector
        psize = np.linalg.norm(p)
        size = np.linalg.norm(self.u)
        
        # calculate angles between p and basis vectors
        # check that total angle is the same as calculated when initializing
        cos_angle_u = np.dot(self.u, p) / (psize * size)
        cos_angle_v = np.dot(self.v, p) / (psize * size)
        tot_angle = np.arccos(cos_angle_u) + np.arccos(cos_angle_v)
        cos_tangle = np.cos(self.angle)
        assert(abs(np.cos(tot_angle) - cos_tangle) < eps)
        
        
        factor = psize / (1 - cos_tangle**2.0)
        a = factor * (cos_angle_u - np.cos(self.angle)*cos_angle_v) / size
        b = factor * (cos_angle_v - np.cos(self.angle)*cos_angle_u) / size
        
        # check the lcs coordinates are within the triagle face
        assert(a >= 0 and b >= 0)
        assert((a + b) <= 1.0)
        return (a, b)
    
    def lcs_to_global(self, point):
        """
        Convert point in lcs to global (3D) coordinates
        """
        p = self.middle + point[0]*self.u + point[1]*self.v
        return p
    
    def lcs_to_net(self, point, v1net, v2net, v3net, scale=1):
        """        
        Convert coordinates in lcs to coordinates on the icosahedron net grid
        
        Parameters
        ----------
        point: array-like
            2D coordinates of a point in the lcs of the face
        v1net, v2net, v3net: array-like
            2D coordinates of the vertices of the face on the net
        scale: int or float
            scale used in the icosahedron net
            default = 1
            
        Returns
        -------
        NumPy Array: coordinates of the point on the net
        """
        v1net = np.array(v1net)
        v2net = np.array(v2net)
        v3net = np.array(v3net)
        
        # corresponding u and v vectors form the face lcs on the net
        unet = v2net - v1net
        vnet = v3net - v1net
        
        # coordinates of the point on the net
        pnet = v1net + point[0]*unet + point[1]*vnet
        return pnet
        

class Icosahedron(object):
    """
    Icosahedron (with 20 faces).
    """
    def __init__(self):
        golden = (1. + np.sqrt(5)) / 2.0

        # All of our verices for the icosahedron
        A = Vertex(golden, 1, 0, "A")
        B = Vertex(-golden, 1, 0, "B")
        C = Vertex(golden, -1, 0, "C")
        D = Vertex(-golden, -1, 0, "D")
        
        E = Vertex(1, 0, golden, "E")
        F = Vertex(-1, 0, golden, "F")
        G = Vertex(1, 0, -golden, "G")
        H = Vertex(-1, 0, -golden, "H")
        
        I = Vertex(0, golden, 1, "I")
        J = Vertex(0, -golden, 1, "J")
        K = Vertex(0, golden, -1, "K")
        L = Vertex(0, -golden, -1, "L")
        
        self.vertices = [A, B, C, D, E, F, G, H, I, J, K, L]
        
        # All of our Faces for the icosahedron
        F1 = Face([H, L, D], 1)
        F2 = Face([H, D, B], 2)
        F3 = Face([H, B, K], 3)
        F4 = Face([H, K, G], 4)
        F5 = Face([H, G, L], 5)
        
        F6 = Face([C, L, G], 6)
        F7 = Face([G, A, C], 7)
        F8 = Face([A, G, K], 8)
        F9 = Face([K, I, A], 9)
        F10 = Face([I, K, B], 10)
        F11 = Face([B, F, I], 11)
        F12 = Face([F, B, D], 12)
        F13 = Face([D, J, F], 13)
        F14 = Face([J, D, L], 14)
        F15 = Face([L, C, J], 15)
        
        F16 = Face([E, J, C], 16)
        F17 = Face([E, C, A], 17)
        F18 = Face([E, A, I], 18)
        F19 = Face([E, I, F], 19)
        F20 = Face([E, F, J], 20)
        
        self.faces = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, 
                      F13, F14, F15, F16, F17, F18, F19, F20]
        
    def pick_face(self, p):
        """
        Pick which face a point p should be projected on
        
        Parameters
        ----------
        p: array-like
            point p in theta, phi coordinates
            
        Returns
        -------
        Face: The face p is to be projected on
        """
        lowest = np.pi*2
        closest_face = self.faces[0]
        
        for f in self.faces:
            pmiddle = point_to_sphere(f.middle)
            dist = ang_distance(p, pmiddle)
            if dist < lowest:
                lowest = dist
                closest_face = f
        
        return closest_face
    
    def project_in_3D(self, p):
        """
        Project a point p onto one of the faces of the icosahedron.
        
        Parameters
        ----------
        p: array-like
            point p in theta, phi coordinates
            
        Returns
        -------
        Face: Face the point was projected on
        NumPy Array: (x, y, z) coodinates of the projected point
            in global coordinates
        """

        face = self.pick_face(p)
        # projected point in the intersection of a ray from the origin through
        # the point in spherical coordinates and the plane of the face
        pcart = sphere_to_cart(p)
        projp = (np.dot(face.middle, face.normal) / 
                 np.dot(pcart, face.normal)) * pcart
        return face, projp
    
    def project_in_lcs(self, p):
        """
        Project a point onto one of the faces of the icosahedron.
        
        Parameters
        ----------
        p: array-like
            point p in theta, phi coordinates
        
        Returns
        -------
        Face: Face the point was projected on
        NumPy Array: (x, y) coordinates of the projected point
            in the lcs of the face
        """
        face, projp3D = self.project_in_3D(p)
        return face, face.global_to_lcs(projp3D)

def ang_distance(p1, p2):
    """
    Calculate angular distance between two point on the 1-sphere p1 and p2.
    
    Parameters
    ----------
    p1: array-like
        point 1 in theta, phi coordinates
    p2: array-like
        point 2 in theta, phi coordinates
        
    Returns
    -------
    float: angular distance between p1, p2 in square radians
    """    
    term1 = np.sin(p1[0]) * np.sin(p2[0]) * np.cos(p1[1] - p2[1])
    term2 = np.cos(p1[0]) * np.cos(p2[0])
    return np.sqrt(2. - 2 * (term1 + term2))

def point_to_sphere(p):
    """
    Projects an input coordinates to the spherical surface.

    Arguments
    ---------
    p: Numpy Array
        x, y, z in Cartesian coordinates

    Returns
    -------
    Numpy Array
        theta, phi
        theta ranges from 0 (North pole) to (and including) pi (South pole)
        phi ranges from 0 (positive x-axis) to (but not including) 2 pi,
        increasing anti-clockwise when looking down on the plane from the top
    """

    r = np.sqrt(np.dot(p, p))
    if r == 0:
        print("Error: r should'nt be 0")
        return None
        
    x = p[0]
    y = p[1]
    z = p[2]

    phi = np.arctan2(y, x)%(2*np.pi)
    theta = np.arccos(z/r)    

    return np.array([theta, phi])


def sphere_to_cart(p, r=1.):
    """
    Transforms point on the sphere to Cartesian coordinates
    
    Arguments
    ---------
    p: Numpy Array
        point (theta, phi) on the sphere
    r: float
        Radius of the sphere
        default: 1.0
    
    Returns
    -------
    Numpy Array
        point (x, y, z) in Cartesian coordinates
    """
    theta = p[0]
    phi = p[1]
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x, y, z])

if __name__=="__main__":
    ICO = Icosahedron()