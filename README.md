# paper-globes
This software projects a spherical image onto the surfaces of a polyhedron, and plots these unto a net that can be cut-out and folded up to make a paper globe.

The software currently is able to build an icosahedron and project points in spherical coordinates onto it. 
A globe is build using coordinates and magnitudes of stars (in ['stars.dat'](stars.dat)).

# Usage
Run the file ['plot_globe.py'](plot_globe.py) to make an Icosahedron paper globe. 
Inside this script, options such as the output file name and colours can be changed.
```
python plot_globe.py
```

# Files
files in this repository are organised as follows
```
.
├── globes                      ## Python scripts
│   ├── import_stars.py         # Import star data
│   ├── projection.py           # Main projection code
├── Icosahedron_net.py          # Building the icosahedron net
├── info                        ## Some useful files and pictures
│   ├── icosahedron_naming.png  # Naming conventions used for the icosahedron
│   ├── notes.txt               # Some notes on the projection and conventions
│   └── paper_globe_pic.jpg     # Picture of the finished paper globe
├── plot_globe.py               # Run scripts
├── stars.dat                   # Star data (coordinates, magnitudes, some names)
└── test-projection.py          # Some tests for projection
```

# Copying
`paper-globes` is licensed under the terms of the MIT License. Please see
the file [`LICENSE`](LICENSE) for full details.
