import math
import matplotlib.pyplot as plt
from globes.projection import Icosahedron
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
               
        # dictionary with net coordinates of all middles per face
        self.mnets = {1:self.node(1, 2.5), 2:self.node(3, 2.5), 3:self.node(5, 2.5),
                      4:self.node(7, 2.5), 5:self.node(9, 2.5), 6:self.node(9, 1.5), 
                      7:self.node(8, 1.5), 8:self.node(7, 1.5), 9:self.node(6, 1.5), 
                      10:self.node(5, 1.5), 11:self.node(4, 1.5), 12:self.node(3, 1.5), 
                      13:self.node(2, 1.5), 14:self.node(1, 1.5), 15:self.node(10, 1.5), 
                      16:self.node(10, 0.5), 17:self.node(8, 0.5), 18:self.node(6, 0.5), 
                      19:self.node(4, 0.5), 20:self.node(2, 0.5)}
        # dictionary with net coordinates of all first vertices per face
        self.unets = {1:self.h1, 2:self.h2, 3:self.h3, 4:self.h4, 5:self.h5, 
                      6:self.node(9, 1), 7:self.node(8, 2), 8:self.node(7, 1), 
                      9:self.node(6, 2), 10:self.node(5, 1), 11:self.node(4, 2), 
                      12:self.node(3, 1), 13:self.node(2, 2), 14:self.node(1, 1), 
                      15:self.l2, 16:self.e5, 17:self.e4, 18:self.e3, 
                      19:self.e2, 20:self.e1}
        
    def node(self, x, y):
        """
        Get positions of a node in the icosahedron net
        Note: for size of a triangle x = 2, then height is y = sqrt(3)
        """
        return [x*self.scale, y*math.sqrt(3)*self.scale]
   
    def plot_line(self, n1, n2, ax):
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
        ax.plot(x, y, 'k-')
        
    def plot_net(self, ax):
        """
        Plot lines of the icosahedron net on ax
        
        Parameters
        ----------
        ax: matplotlib.axis
        """
        self.plot_line(self.l1, self.l2, ax)
        self.plot_line(self.j1, self.j2, ax)
        self.plot_line(self.l1, self.e1, ax)
        self.plot_line(self.h1, self.e2, ax)
        self.plot_line(self.h2, self.e3, ax)
        self.plot_line(self.h3, self.e4, ax)
        self.plot_line(self.h4, self.e5, ax)
        self.plot_line(self.h5, self.j2, ax)
        self.plot_line(self.l1, self.h1, ax)
        self.plot_line(self.j1, self.h2, ax)
        self.plot_line(self.e1, self.h3, ax)
        self.plot_line(self.e2, self.h4, ax)
        self.plot_line(self.e3, self.h5, ax)
        self.plot_line(self.e4, self.l2, ax)
        self.plot_line(self.e5, self.j2, ax)

    def plot_star(self, star, ax):
        """
        Project star onto icosahedron face, then plot on net (on ax)
        """
        face, proj_star = self.Ico.project_in_lcs([star.dec, star.ra])
        if face.ID == 12:
            print proj_star
            net_star = face.lcs_to_net(proj_star, self.mnets[face.ID], 
                                       self.unets[face.ID], self.scale)
        
            ax.scatter(net_star[0], net_star[1], c='k', s=2*star.mag, marker='*')
        else:
            pass       
        
    
    def make_globe(self, stars=True):
        """
        Create a figure, plot the net, and plot stars
        """
        # set up figure with the right size
        xsize = self.j2[0] - self.l1[0]
        ysize = self.h1[1] - self.e1[1]
        
        fig = plt.figure(figsize=(xsize*self.scale, ysize*self.scale))
        ax = fig.add_subplot(111)
        ax.set_xlim([0, xsize])
        ax.set_ylim([0, ysize])
        
        self.plot_net(ax)
        
        if stars:
            stars = get_stars()
            for s in stars:
                self.plot_star(s, ax)
                
        ax.scatter(self.mnets[12][0], self.mnets[12][1], c='r', s=6)
        ax.scatter(self.unets[12][0], self.unets[12][1], c='g', s=30)
        
        ax.axis('off')
        fig.savefig('paper_globe.pdf', bbox_inches='tight')

if __name__=="__main__":
    Iconet = IcosahedronNet()
    Iconet.make_globe(stars=True)

