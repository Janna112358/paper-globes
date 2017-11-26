#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from globes.projection import Icosahedron
from globes.import_stars import get_stars

class IcosahedronNet():
    """
    Net, based on nodes in a 2D coordinate system
    """
    def __init__(self, scale=1):
        """
        Parameters
        ----------
        Ico: Ico
            Icosahedron
        scale: int or float
            scale for the figure, triangel side is 2*scale
            default = 1
        """
        self.Ico = Icosahedron()
        self.scale = scale
        
        # create notes of the icosahedron net
        self.l1 = self.node(0, 2)
        self.l2 = self.node(10, 2)
        self.j1 = self.node(1, 1)
        self.j2 = self.node(11, 1)
        self.h1 = self.node(1, 3)
        self.h2 = self.node(3, 3)
        self.h3 = self.node(5, 3)
        self.h4 = self.node(7, 3)
        self.h5 = self.node(9, 3)
        self.e1 = self.node(2, 0)
        self.e2 = self.node(4, 0)
        self.e3 = self.node(6, 0)
        self.e4 = self.node(8, 0)
        self.e5 = self.node(10, 0)
        
        self.d = self.node(2, 2)
        self.b = self.node(4, 2)
        self.k = self.node(6, 2)
        self.g = self.node(8, 2)
        self.f = self.node(3, 1)
        self.i = self.node(5, 1)
        self.a = self.node(7, 1)
        self.c = self.node(9, 1)
               
        # dictionary with net coordinates of all middles per face
        self.mnets = {1:self.node(1, 2.5), 2:self.node(3, 2.5), 3:self.node(5, 2.5),
                      4:self.node(7, 2.5), 5:self.node(9, 2.5), 6:self.node(9, 1.5), 
                      7:self.node(8, 1.5), 8:self.node(7, 1.5), 9:self.node(6, 1.5), 
                      10:self.node(5, 1.5), 11:self.node(4, 1.5), 12:self.node(3, 1.5), 
                      13:self.node(2, 1.5), 14:self.node(1, 1.5), 15:self.node(10, 1.5), 
                      16:self.node(10, 0.5), 17:self.node(8, 0.5), 18:self.node(6, 0.5), 
                      19:self.node(4, 0.5), 20:self.node(2, 0.5)}
        # dictionary with net coordinates of all first vertices per face
        self.v1nets = {1:self.h1, 2:self.h2, 3:self.h3, 4:self.h4, 5:self.h5, 
                      6:self.c, 7:self.g, 8:self.a, 9:self.k, 10:self.i, 
                      11:self.b, 12:self.f, 13:self.d, 14:self.j1, 15:self.l2, 
                      16:self.e5, 17:self.e4, 18:self.e3, 19:self.e2, 20:self.e1}
        # dictionary with net coordinates of all second vertices per face
        self.v2nets = {1:self.l1, 2:self.d, 3:self.b, 4:self.k, 5:self.g, 
                      6:self.l2, 7:self.a, 8:self.g, 9:self.i, 10:self.k,
                      11:self.f, 12:self.b, 13:self.j1, 14:self.d, 15:self.c,
                      16:self.j2, 17:self.c, 18:self.a, 19:self.i, 20:self.f}
        # dictionary with net coordinates of all third vertices per face
        self.v3nets = {1:self.d, 2:self.b, 3:self.k, 4:self.g, 5:self.l2, 
                      6:self.g, 7:self.c, 8:self.k, 9:self.a, 10:self.b,
                      11:self.i, 12:self.d, 13:self.f, 14:self.l1, 15:self.j2,
                      16:self.c, 17:self.a, 18:self.i, 19:self.f, 20:self.j1}
        
    def node(self, x, y):
        """
        Get positions of a node in the icosahedron net
        Note: for size of a triangle x = 2, then height is y = sqrt(3)
        """
        return [x*self.scale, y*np.sqrt(3)*self.scale]
   
    def plot_line(self, n1, n2, ax, c='k', ls='-', label=None):
        """
        Plot line between nodes n1-n2 on ax
        
        Parameters
        ----------
        n1, n2: array-like
            start and end nodes of the line to plot
        ax: matplotlib.axis
            axis to plot on
        c: matplotlib colour (str)
            default = 'k' (black)
        ls: matplotlib linestyle (str)
            default = '-' (solid)
        label: str
            default = None 
        """
        x = [n1[0], n2[0]]
        y = [n1[1], n2[1]]
        ax.plot(x, y, c=c, ls=ls, label=label)
        
    def plot_net(self, ax, c='k', fold_ls = '--', cut_ls = '-', label=None):
        """
        Plot lines of the icosahedron net on ax
        
        Parameters
        ----------
        ax: matplotlib.axis
        c: matplotlib color (str)
            default = 'k' (black)
        fold_ls: matplotlib linestyle (str)
            linestyle for net lines that are to be folded
            default = '--' (dashed)
        cut_ls: matplotlib linestyle (str)
            linestyle for net lines that are to be cut 
            (outer lines without glue band)
            default = '-' (solid)
        label: str
            default = None
        """
        self.plot_line(self.l1, self.l2, ax, c=c, ls=fold_ls, label=label)
        self.plot_line(self.j1, self.j2, ax, c=c, ls=fold_ls)
        self.plot_line(self.l1, self.e1, ax, c=c, ls=fold_ls)
        self.plot_line(self.h1, self.e2, ax, c=c, ls=fold_ls)
        self.plot_line(self.h2, self.e3, ax, c=c, ls=fold_ls)
        self.plot_line(self.h3, self.e4, ax, c=c, ls=fold_ls)
        self.plot_line(self.h4, self.e5, ax, c=c, ls=fold_ls)
        self.plot_line(self.h5, self.j2, ax, c=c, ls=cut_ls)
        self.plot_line(self.l1, self.h1, ax, c=c, ls=fold_ls)
        self.plot_line(self.j1, self.d, ax, c=c, ls=fold_ls)
        self.plot_line(self.d, self.h2, ax, c=c, ls=cut_ls)
        self.plot_line(self.e1, self.f, ax, c=c, ls=cut_ls)
        self.plot_line(self.f, self.b, ax, c=c, ls=fold_ls)
        self.plot_line(self.b, self.h3, ax, c=c, ls=cut_ls)
        self.plot_line(self.e2, self.i, ax, c=c, ls=cut_ls)
        self.plot_line(self.i, self.k, ax, c=c, ls=fold_ls)
        self.plot_line(self.k, self.h4, ax, c=c, ls=cut_ls)
        self.plot_line(self.e3, self.a, ax, c=c, ls=cut_ls)
        self.plot_line(self.a, self.g, ax, c=c, ls=fold_ls)
        self.plot_line(self.g, self.h5, ax, c=c, ls=cut_ls)
        self.plot_line(self.e4, self.c, ax, c=c, ls=cut_ls)
        self.plot_line(self.c, self.l2, ax, c=c, ls=fold_ls)
        self.plot_line(self.e5, self.j2, ax, c=c, ls=cut_ls)
    
    def plot_connected_lines(self, nodes, ax, c='k', ls='-', label=None):
        """
        Plot connected lines on ax from the first node in lines to the last.
        
        Parameters
        ----------
        nodes: list of nodes
            each node is a 2D array-like object with coordinates on the net.
        ax: matplotlib.axis
        c: matplotlib colour (str)
            default = 'k' (black)
        ls: matplotlib linestyle (str)
            default = '-' (solid)
        label: str
            default = None
        """
        for i in range(len(nodes) - 1):
            if i == 0:
                l = label
            else:
                l = None
            self.plot_line(nodes[i], nodes[i+1], ax, 
                           c=c, ls=ls, label=l)
        
    def plot_glue_bands(self, ax, w=0.4, c='k', ls='-', label=None):
        """
        Plot lines for glue edges for the icosahedron net on ax
        
        Parameters
        ----------
        ax: matplotlib.axis
        w: float
            size scale for the glue edges
            default = 0.2
        c: matplotlib colour (str)
            default = 'k' (black)
        ls: matplotlib linestyle (str)
            default = '-' (solid)
        label: str
            default = None
        """
        wx = 0.5 * w
        wy = 0.5 * w
        eps = 0.03
             
        j1_w1 = self.node(1-w, 1)
        l1_w2 = self.node(0-wx, 2-wy)
        self.plot_connected_lines([self.j1, j1_w1, l1_w2, self.l1], ax, 
                                  c=c, ls=ls, label=label)
        
        e1_w = self.node(2-w, 0)
        j1_w2 = self.node(1-wx, 1-wy)
        self.plot_connected_lines([self.e1, e1_w, j1_w2, self.j1], ax, 
                                  c=c, ls=ls)
        
        e2_w = self.node(4-w, 0)
        f_w = self.node(3-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e2, e2_w, f_w, self.f], ax, 
                                  c=c, ls=ls)
        
        e3_w = self.node(6-w, 0)
        i_w = self.node(5-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e3, e3_w, i_w, self.i], ax, 
                                  c=c, ls=ls)
        
        e4_w = self.node(8-w, 0)
        a_w = self.node(7-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e4, e4_w, a_w, self.a], ax, 
                                  c=c, ls=ls)
        
        e5_w = self.node(10-w, 0)
        c_w = self.node(9-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e5, e5_w, c_w, self.c], ax, 
                                  c=c, ls=ls)
        
        # double edge on this triangle for easier glueing
        l1_w1 = self.node(0-wx, 2+wy)
        h1_w1 = self.node(1-w, 3)
        self.plot_connected_lines([self.l1, l1_w1, h1_w1, self.h1], ax, 
                                  c=c, ls=ls)
        
        h1_w2 = self.node(1+w, 3)
        d_w = self.node(2+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h1, h1_w2, d_w, self.d], ax, 
                                  c=c, ls=ls)
        
        h2_w = self.node(3+w, 3)
        b_w = self.node(4+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h2, h2_w, b_w, self.b], ax, 
                                  c=c, ls=ls)
        
        h3_w = self.node(5+w, 3)
        k_w = self.node(6+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h3, h3_w, k_w, self.k], ax, 
                                  c=c, ls=ls)
        
        h4_w = self.node(7+w, 3)
        g_w = self.node(8+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h4, h4_w, g_w, self.g], ax, 
                                  c=c, ls=ls)
        
        # leave out edge on h5 face, sine the h1 face has both
          
        
    def plot_point(self, point, ax, zorder=0, c='k', s=2, marker='*', 
                   label=None, text=None, xoff=0., yoff=0., rot=0, textc=None, texts=10):
        """
        Project a point onto an icosahedron surface, then plot on the net.
        
        Parameters
        ----------
        point: array-like
            2D theta, phi coordinates of a point on the spherical image 
            (e.g. a star's coordinates), or array of points
        ax: matplotlib.axis
            axis to plot on
        zorder: order for matplotlib to plot things in, lowest first
            default = 0
        c: matplotlib color
            can be array of colors, needs to be the same length as points
            default = 'k' (black)
        s: int or float, or array-like
            markersize, scaled with Icosahedronnet.scale
            if array-like, needs to be the same length as points
            default = 2
        marker: matplotlib marker
            marker for the point to plot
            can be array of markers, needs to be same length as points
            default: '*'
        label: str
            default = None
        text: str
            text to be plotted close to the point
            default = None
        xoff: number
            for text, displacement from point in x-direction
            default = 0
        yoff: number
            for text, displace from point in y-direction
            default = 0
        rot: number
            for text, rotation of the text w.r.t. horizontal, in degrees
            default = 0
        textc: matplotlib colour or None
            fot text, colour. If None, use same colour as points
            default = None
        texts: int
            for text, fontsize
            default = 10
        """
        ms = np.array(s) * self.scale
        
        if np.array(point).ndim == 1:
            face, projp = self.Ico.project_in_lcs(point)
            net_point = face.lcs_to_net(projp, self.v1nets[face.ID], 
                                    self.v2nets[face.ID], self.v3nets[face.ID])
            net_points = net_point
        
        else:
            net_points = np.zeros(point.shape)
            for i, p in enumerate(point):
                face, projp = self.Ico.project_in_lcs(p)
                net_point = face.lcs_to_net(projp, self.v1nets[face.ID], 
                                    self.v2nets[face.ID], self.v3nets[face.ID])
                net_points[i] = net_point
                
        ax.scatter(np.take(net_points, 0, axis=-1), np.take(net_points, 1, axis=-1),
                   c=c, s=ms, marker=marker, label=label, zorder=zorder)
        if text is not None:
            if textc is None:
                textc = c
            ax.text(net_points[0]+xoff, net_points[1]+yoff, text, 
                    color=textc, rotation=rot, fontsize=texts, zorder=zorder)
    
    def make_globe(self, stars=True, poles=True, dataDir=None, 
                   fname='paper_globe.pdf', edge_width=0.4, 
                   starc='w', linec = 'k', bgc='darkblue', bgalpha=1.0, texts=12):
        """
        Create a figure, plot the net, and plot stars
        
        Parameters
        ----------
        stars: bool
            project and plot stars or not
            default = True
        poles:
            plot dots for north and south pole or not
            default = True
        dataDir: str or none
            path to the directory with stars.dat
            if None, assumes stars.dat in the working directory
        fname: str
            filename for saving the paper globe
            default = 'paper_globe.pdf'
        edge_width: float
            width for glue edged of the net, scales with net scale
            default = 0.4
        starc: matplotlib colour
            colour for stars
            default: 'w' (white)
        linec: matplotlib colour
            colour for net lines and glue bands
            default: 'k' (black)
        bgc: matplotlib colour
            colour for background
            default: 'darkblue'
        texts: int
            fontsize for any text on labelled points
        """    
        # set up figure with the right size
        xspan = (self.j2[0] - self.l1[0])
        yspan = (self.h1[1] - self.e1[1])
        xsize = 1.02 * xspan
        ysize = 1.02 * yspan
        
        fig = plt.figure(figsize=(xsize*self.scale, ysize*self.scale))
        ax = fig.add_subplot(111)
        ax.set_xlim([-0.55*edge_width*self.scale, xsize])
        ax.set_ylim([0, ysize])
        # draw background first, so lowest zorder
        ax.axhspan(-0.01, yspan+0.01, xmin=-0.01, xmax=xspan/xsize+0.002, 
                   facecolor=bgc, alpha=bgalpha, zorder=0)
        
        # Plot icosahedron net and glue bands
        self.plot_net(ax, c=linec, fold_ls='--', cut_ls='-', label = 'Fold me')
        self.plot_glue_bands(ax, w=edge_width, c=linec, ls='-', label='Cut me')
        
        # text for last triangle to glue
        glue_text_pos = self.node(9.3, 2.8)
        ax.text(glue_text_pos[0], glue_text_pos[1], 'Glue me last!', 
                rotation=-60, color=starc, fontsize=14)
        
        # generic text
        title_text_pos = self.node(0.0, 1.0)
        ax.text(title_text_pos[0], title_text_pos[1], 'Galaxy paper globe', 
                rotation=-60, color=starc, fontsize=20)
        ax.text(0.0, 0.1, 'github.com/Janna112358/paper-globes',
                color=starc, fontsize=12)
        
        # plot north and south poles as dots
        if poles:
            north = [0.0, 0.0]
            south = [np.pi, 0.0]
            self.plot_point(north, ax, c='r', s=20, marker='o', 
                            label='North', zorder=1)
            self.plot_point(south, ax, c='y', s=20, marker='o', 
                            label='South', zorder=1)
        
        # Load, project and plot stars
        if stars:
            stars = get_stars(dataDir=dataDir)
            for s in stars:
                # largest stars with star marker, others with dot
                if s.mag < 2.5:
                    m = '*'
                    size=30*np.exp(-s.mag)
                else:
                    m = 'o'
                    size=30*np.exp(-s.mag)
                
                # plot names for some stars
                if s.name == 'Sirius':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=0.02, yoff=0.02, rot=60, texts=texts)
                elif s.name == 'Vega':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.25, yoff=0.1, rot=240, texts=texts)
                elif s.name == 'Canopus':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.05, yoff=0.05, rot=60, texts=texts)
                elif s.name=='Alpha Centauri':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.26, yoff=0.14, rot=-60, texts=texts)
                elif s.name=='Arcturus':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.22, yoff=0.12, rot=-60, texts=texts)
                elif s.name=='Capella':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.24, yoff=0.14, rot=240, texts=texts)
                elif s.name=='Rigel':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=0., yoff=0., rot=60, texts=texts)
                elif s.name=='Procyon':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.2, yoff=0.05, rot=300, texts=texts)
                elif s.name=='Achernar':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.2, yoff=0., rot=300, texts=texts)
                elif s.name=='Betelgeuse':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=-0.14, yoff=0.14, rot=60, texts=texts)
                elif s.name=='Polaris':
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=0., yoff=0.1, rot=120, texts=texts)
                elif s.name is not None:
                    self.plot_point([s.dec, s.ra], ax, zorder=2,
                                    c=starc, s=size, marker=m, text=s.name, 
                                    xoff=0., yoff=0., rot=60, texts=texts)
                # all cases without names
                else:
                    self.plot_point([s.dec, s.ra], ax, c=starc, s=size, 
                                marker=m, zorder=2) 

        ax.legend(loc=[0.9, 0.8], fontsize=14)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(fname, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
        
    def test_points(self, pointc = 'k', linec='k', bgc='w'):
        """
        Plot a net with some test points
        
        pointc: matplotlib colour (str)
            colour for the points
            default = 'k' (black)
        linec: matplotlib colour (str)
            colour for the lines (of the net)
            default = 'k' (black)
        bgc: matplotlib colour (str)
            background colour
            default = 'w' (white)
        """
        
        # set up figure with the right size
        xsize = self.j2[0] - self.l1[0]
        ysize = self.h1[1] - self.e1[1]
        
        fig = plt.figure(figsize=(xsize*self.scale, ysize*self.scale))
        ax = fig.add_subplot(111)
        ax.set_xlim([0, xsize])
        ax.set_ylim([0, ysize])
        
        # plot net and glue bands
        self.plot_net(ax, c=linec, ls='--')
        self.plot_glue_band(ax, c=linec, ls='-')
        
        ax.scatter(self.v1nets[12][0], self.v1nets[12][1], c='r', s=30)
        ax.scatter(self.v2nets[12][0], self.v2nets[12][1], c='g', s=30)
        ax.scatter(self.v3nets[12][0], self.v3nets[12][1], c='b', s=30)
        
        # list some points for testing, project them onto the icosahedron
        points = [[0.1 * np.pi * j, 0.3 * np.pi] for j in range(10)]
        pface = []
        pproj = []
        for i, p in enumerate(points):
            f, pp = self.Ico.project_in_lcs(p)
            pface.append(f)
            pproj.append(pp)
            
        # plot projected point onto the net
        for j, p in enumerate(pproj):
            face = pface[j]
            net_point = face.lcs_to_net(p, self.v1nets[face.ID], 
                        self.v2nets[face.ID], self.v3nets[face.ID])
            ax.scatter(net_point[0], net_point[1], c='k', marker='*')
        fig.savefig('test_points.pdf', bbox_inches='tight')

if __name__=="__main__":
    Iconet = IcosahedronNet(scale=1.5)
    Iconet.make_globe(stars=True, dataDir='.')

