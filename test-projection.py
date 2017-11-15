#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:18:27 2017

@author: jgoldstein
"""
import numpy as np
import globes.projection as projection

def middles_test(ICO):
    """
    Test to check whether projected middles of faces get projected back
    onto the original middles.
    
    Parameters
    ----------
    ICO: projection.Icosahedron
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
        print("Middles test succesfull, middles projected back onto middles")
    else:
        print("Middles test failed, middles projected back elsewhere than" +
              "original middles")
        
        
if __name__ == "__main__":
    ICO = projection.Icosahedron()
    middles_test(ICO)
    