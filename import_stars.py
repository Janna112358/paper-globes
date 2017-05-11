import csv
import math

with open('./stars.dat', 'rb') as f:
    data = list(csv.reader(f))

ra_frac = []
dec_frac = []

for i, e in enumerate(data):

    ra = e[0]
    dec = e[1]
    mag = e[2]

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

    ra_frac.append(math.radians(ra))
    dec_frac.append(math.radians(dec) + math.pi/2.)
