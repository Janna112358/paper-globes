#!/usr/bin/env python
from collections import namedtuple
import csv
import numpy as np
import os
import sys


def get_stars(dataDir=None):
    # Get path to stars.dat and check whether is exists
    if dataDir is None:
        path = './stars.dat'
    else:
        path = os.path.join(dataDir, 'stars.dat')
    try:
        assert(os.path.exists(path))
    except AssertionError:
        print("[error] Path to stars.dat does not exist: " + path)
        sys.exit(1)
        
    # Import data
    with open(path, 'r') as f:
        data = list(csv.reader(f))

    Star = namedtuple('Star', 'ra dec mag name')
    stars = []

    for i, e in enumerate(data):
        
        ra = e[0]
        dec = e[1]
        m = float(e[2])
        
        if dec[1] == ' ':
            dec = dec.replace(' ', '', 1)

        rh, rm, rs = [float(r) for r in ra.split(' ')]
        # right ascension in degrees (times 360 deg/ 24 hours)
        ra = rh*15 + rm/4 + rs/240
        dd, dm, ds = [float(d) for d in dec.split(' ')]
        if dd < 0:
            sign = -1
        else:
            sign = 1
        # declination in degrees
        dec = dd + sign*dm/60. + sign*ds/3600.

        # convert to theta, phi coordinates
        ra_frac = np.radians(ra)
        dec_frac = np.pi/2. - np.radians(dec)
        
        try:
            name = e[3].strip()
        except:
            name=None
        
        s = Star(ra=ra_frac, dec=dec_frac, mag=m, name=name)
        stars.append(s)
        
    return stars
        
if __name__=="__main__":
    stars = get_stars(dataDir = '..')