#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Icosahedron_net import IcosahedronNet

### Settings for your paper globe ###
stars_directory = '.'           # (relative) path from this script to the stars.dat file
filename = 'paper_globe.pdf'    # filename for your paper globe
scale = 1.5                     # overall size scale (changing this messes up text fontsize)
bgcolour = 'darkblue'           # background colour
starcolour = 'w'                # colour of the stars
linecolour = 'k'                # colour of the lines of the net
# other colours can be found at e.g. https://matplotlib.org/examples/color/named_colors.html
bgalpha = 1.0                   # seethroughness of the background (0 is invisible, 1 is opaque)
textsize = 12                   # size for the star labels
# for more options, see documentation in Icosahedron_net.py

if __name__ == "__main__":
    Iconet = IcosahedronNet(scale=scale)
    Iconet.make_globe(stars=True, dataDir='.', fname=filename,
                      bgc=bgcolour, starc=starcolour, linec = linecolour, 
                      bgalpha=bgalpha, texts=textsize)

