#!/usr/bin/env python


class Surface(object):

    def __init__(self):
        pass


class Vertex(object):

    def __init__(self, x, y, z, name):
        """
        Creates the Vertex Object with Carthesian coordinates
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param name: The name (letter) of the Vertex
        :return: Void
        """
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def to_spherical(self):
        """
        Maybe we'll need it, who knows?
        :return:
        """
        pass

