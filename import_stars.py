from collections import namedtuple
import csv
import math

# Import data
with open('./stars.dat', 'rb') as f:
    data = list(csv.reader(f))

Star = namedtuple('Star', 'ra dec mag')
stars = []

for i, e in enumerate(data):

    ra = e[0]
    dec = e[1]
    m = float(e[2])

    if dec[1] == ' ':
        dec = dec.replace(' ', '', 1)

    rh, rm, rs = [float(r) for r in ra.split(' ')]
    ra = rh*15 + rm/4 + rs/240
    dd, dm, ds = [float(d) for d in dec.split(' ')]
    if dd < 0:
        sign = -1
    else:
        sign = 1
    dec = dd + sign*dm/60 + sign*ds/3600

    ra_frac = math.radians(ra)
    dec_frac = math.radians(dec) + math.pi/2.

    s = Star(ra=ra_frac, dec=dec_frac, mag=m)
    stars.append(s)