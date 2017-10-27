import numpy as np
import matplotlib.pyplot as plt
from globes.projection2 import Icosahedron
from import_stars import get_stars

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
   
    def plot_line(self, n1, n2, ax, linestyle='-', label=None):
        """
        Plot line between nodes n1-n2 on ax
        
        Parameters
        ----------
        n1, n2: array-like
            start and end nodes of the line to plot
        ax: matplotlib.axis
            axis to plot on
        """
        x = [n1[0], n2[0]]
        y = [n1[1], n2[1]]
        ax.plot(x, y, c='k', ls=linestyle, label=label)
        
    def plot_net(self, ax, linestyle='--', label=None):
        """
        Plot lines of the icosahedron net on ax
        
        Parameters
        ----------
        ax: matplotlib.axis
        """
        self.plot_line(self.l1, self.l2, ax, linestyle=linestyle, label=label)
        self.plot_line(self.j1, self.j2, ax, linestyle=linestyle)
        self.plot_line(self.l1, self.e1, ax, linestyle=linestyle)
        self.plot_line(self.h1, self.e2, ax, linestyle=linestyle)
        self.plot_line(self.h2, self.e3, ax, linestyle=linestyle)
        self.plot_line(self.h3, self.e4, ax, linestyle=linestyle)
        self.plot_line(self.h4, self.e5, ax, linestyle=linestyle)
        self.plot_line(self.h5, self.j2, ax, linestyle=linestyle)
        self.plot_line(self.l1, self.h1, ax, linestyle=linestyle)
        self.plot_line(self.j1, self.h2, ax, linestyle=linestyle)
        self.plot_line(self.e1, self.h3, ax, linestyle=linestyle)
        self.plot_line(self.e2, self.h4, ax, linestyle=linestyle)
        self.plot_line(self.e3, self.h5, ax, linestyle=linestyle)
        self.plot_line(self.e4, self.l2, ax, linestyle=linestyle)
        self.plot_line(self.e5, self.j2, ax, linestyle=linestyle)
    
    def plot_connected_lines(self, nodes, ax, linestyle='-', label=None):
        """
        Plot connected lines on ax from the first node in lines to the last
        """
        for i in range(len(nodes) - 1):
            if i == 0:
                l = label
            else:
                l = None
            self.plot_line(nodes[i], nodes[i+1], ax, 
                           linestyle=linestyle, label=l)
        
    def plot_glue_bands(self, ax, w=0.4, linestyle='-', label=None):
        """
        Plot lines for glue edges for the icosahedron net on ax
        
        Parameters
        ----------
        ax: matplotlib.axis
        
        w: float
            size scale for the glue edges
            default = 0.2
        """
        wx = 0.5 * w
        wy = 0.5 * w
        eps = 0.03
             
        j1_w1 = self.node(1-w, 1)
        l1_w2 = self.node(0-wx, 2-wy)
        self.plot_connected_lines([self.j1, j1_w1, l1_w2, self.l1], ax, 
                                  linestyle=linestyle, label=label)
        
        e1_w = self.node(2-w, 0)
        j1_w2 = self.node(1-wx, 1-wy)
        self.plot_connected_lines([self.e1, e1_w, j1_w2, self.j1], ax, 
                                  linestyle=linestyle)
        
        e2_w = self.node(4-w, 0)
        f_w = self.node(3-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e2, e2_w, f_w, self.f], ax, 
                                  linestyle=linestyle)
        
        e3_w = self.node(6-w, 0)
        i_w = self.node(5-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e3, e3_w, i_w, self.i], ax, 
                                  linestyle=linestyle)
        
        e4_w = self.node(8-w, 0)
        a_w = self.node(7-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e4, e4_w, a_w, self.a], ax, 
                                  linestyle=linestyle)
        
        e5_w = self.node(10-w, 0)
        c_w = self.node(9-wx+eps, 1-wy-eps)
        self.plot_connected_lines([self.e5, e5_w, c_w, self.c], ax, 
                                  linestyle=linestyle)
        
        # double edge on this triangle for easier glueing
        l1_w1 = self.node(0-wx, 2+wy)
        h1_w1 = self.node(1-w, 3)
        self.plot_connected_lines([self.l1, l1_w1, h1_w1, self.h1], ax, 
                                  linestyle=linestyle)
        
        h1_w2 = self.node(1+w, 3)
        d_w = self.node(2+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h1, h1_w2, d_w, self.d], ax, 
                                  linestyle=linestyle)
        
        h2_w = self.node(3+w, 3)
        b_w = self.node(4+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h2, h2_w, b_w, self.b], ax, 
                                  linestyle=linestyle)
        
        h3_w = self.node(5+w, 3)
        k_w = self.node(6+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h3, h3_w, k_w, self.k], ax, 
                                  linestyle=linestyle)
        
        h4_w = self.node(7+w, 3)
        g_w = self.node(8+wx-eps, 2+wy+eps)
        self.plot_connected_lines([self.h4, h4_w, g_w, self.g], ax, 
                                  linestyle=linestyle)
        
        # leave out edge on h5 face, sine the h1 face has both
          
        
    def plot_point(self, point, ax, c='k', s=2, marker='*'):
        """
        Project a point onto an icosahedron surface, then plot on the net.
        
        Parameters
        ----------
        point: array-like
            2D theta, phi coordinates of a point on the spherical image 
            (e.g. a star's coordinates), or array of points
        ax: matplotlib.axis
            axis to plot on
        c: matplotlib color
            can be array of colors, needs to be the same length as points
        s: int or float, or array-like
            markersize, scaled with Icosahedronnet.scale
            if array-like, needs to be the same length as points
            default=2
        marker: matplotlib marker
            marker for the point to plot
            can be array of markers, needs to be same length as points
            default: '*'
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
                
        ax.scatter(net_points[0], net_points[1], c=c, s=ms, marker=marker)
    
    def make_globe(self, stars=True, edge_width=0.4, poles=True):
        """
        Create a figure, plot the net, and plot stars
        """
        # set up figure with the right size
        xsize = 1.05 * (self.j2[0] - self.l1[0])
        ysize = 1.05 * (self.h1[1] - self.e1[1])
        
        fig = plt.figure(figsize=(xsize*self.scale, ysize*self.scale))
        fig.patch.set_facecolor('darkblue')
        ax = fig.add_subplot(111)
        ax.set_xlim([-0.55*edge_width*self.scale, xsize])
        ax.set_ylim([0, ysize])
        
        self.plot_net(ax, linestyle='--', label = 'Fold me')
        self.plot_glue_bands(ax, w=edge_width, linestyle='-', label='Cut me')
        ax.legend(loc='best')
        
        # text for last triangle to glue
        glue_text_pos = self.node(9.3, 2.8)
        ax.text(glue_text_pos[0], glue_text_pos[1], 'Glue me last!', 
                rotation=-60, color='w')
        
        if stars:
            stars = get_stars()
            for s in stars:
                if s.mag < 2:
                    m = '*'
                    size=30*np.exp(-s.mag)
                else:
                    m = 'o'
                    size=30*np.exp(-s.mag)
                self.plot_point([s.dec, s.ra], ax, c='w', s=size, marker=m) 
                
        if poles:
            north = [0.0, 0.0]
            south = [np.pi, 0.0]
            self.plot_point(north, ax, c='r', s=20, marker='o')
            self.plot_point(south, ax, c='y', s=20, marker='o')
        
        ax.axis('off')
        fig.savefig('paper_globe.pdf', bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
        
    def test_points(self):
        
        # set up figure with the right size
        xsize = self.j2[0] - self.l1[0]
        ysize = self.h1[1] - self.e1[1]
        
        fig = plt.figure(figsize=(xsize*self.scale, ysize*self.scale))
        ax = fig.add_subplot(111)
        ax.set_xlim([0, xsize])
        ax.set_ylim([0, ysize])
        
        self.plot_net(ax)
        
        ax.scatter(self.v1nets[12][0], self.v1nets[12][1], c='r', s=30)
        ax.scatter(self.v2nets[12][0], self.v2nets[12][1], c='g', s=30)
        ax.scatter(self.v3nets[12][0], self.v3nets[12][1], c='b', s=30)
        
        points = [[0.1 * np.pi * j, 0.3 * np.pi] for j in range(10)]
        pface = []
        pproj = []
        for i, p in enumerate(points):
            f, pp = self.Ico.project_in_lcs(p)
            pface.append(f)
            pproj.append(pp)
            

        for j, p in enumerate(pproj):
            face = pface[j]
            net_point = face.lcs_to_net(p, self.v1nets[face.ID], 
                        self.v2nets[face.ID], self.v3nets[face.ID])
            print net_point
            ax.scatter(net_point[0], net_point[1], c='k', marker='*')
        fig.savefig('test_points.pdf', bbox_inches='tight')


if __name__=="__main__":
    Iconet = IcosahedronNet(scale=1.5)
    Iconet.make_globe(stars=True)
    #Iconet.test_points()

