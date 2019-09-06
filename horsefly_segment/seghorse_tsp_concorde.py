import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('text', usetex=True)

import matplotlib as mpl
import sys, os, time
import itertools
from colorama import Fore, Style
import utils_graphics
from CGAL.CGAL_Kernel import Point_2, Segment_2
from CGAL.CGAL_Kernel import do_intersect, intersection

from concorde.tsp import TSPSolver
import utils_algo

class SegmentHorsefly:
      def __init__(self, sites=[], inithorseposn=[]):
           self.sites                = sites
           self.inithorseposn        = inithorseposn


      def clearAllStates (self):
          self.sites           = []
          self.inithorseposn   = None

      def getTour(self, fig, ax):
          # Here we assume that the truck AND the drone have the same speed 1.0
          # for this case, the tour follows the reflection TSP
          # Insert meta-data at the top of the file of tsplib file
            
          sites_sorted_by_x = sorted(self.sites , key=lambda k: k[0])
          
          f= open("seghorse.tsp","w+")
          
          f.write("NAME: seghorse\n")
          f.write("TYPE: TSP\n")
          f.write("COMMENT: Segment horsefly problem\n")
          f.write("DIMENSION: " + str(len(sites_sorted_by_x)) + "\n")
          f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
          f.write("EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
          f.write("EDGE_WEIGHT_SECTION\n")

          for i in range(len(sites_sorted_by_x)):
                for j in range(i+1, len(sites_sorted_by_x)):
                      refl_pt, refl_dist = calculate_reflection_point_and_distance(sites_sorted_by_x[i], sites_sorted_by_x[j])
                      
                      # Distances are scaled by 1000 and rounded to an integer
                      # concorde only seems to accept integer values distances
                      # so we scale a largish number to scale all distances and 
                      # discard the fractional part, equal to floor for positive 
                      # numbers
                      scaling_factor = 10000000 # arbitrarily chosen large number
                      f.write(str(int(scaling_factor * refl_dist))+"\n")
                      print refl_pt, refl_dist
          f.write("EOF")
          f.close() 
          
          # Instantiate solver
          solver = TSPSolver.from_tspfile("seghorse.tsp")

          # Find tour
          tour_data = solver.solve()
          assert tour_data.success
          print tour_data.tour
          #solution = state_capitals.iloc[tour_data.tour]
          
          # Now plot the tour of the horse and the fly
          # and  place numbered patches on the sites according to the order visited
          k = 1
          optimal_tour_length = 0.0
          for i,j in zip(tour_data.tour, tour_data.tour[1:]):
                 p = sites_sorted_by_x[i]
                 q = sites_sorted_by_x[j]

                 refl_pt, refl_dist = calculate_reflection_point_and_distance(p,q)

                 xs = [ p[0], refl_pt[0], q[0] ] 
                 ys = [ p[1], refl_pt[1], q[1] ]
                 
                 ax.plot(xs,ys,'g-')
                 ax.text(p[0], p[1], str(k), horizontalalignment='center', verticalalignment='center',
                         fontsize=20, bbox=dict(facecolor='#ddcba0', alpha=1.0, pad=3.0)) 
     
                 k += 1
                 optimal_tour_length += np.linalg.norm( np.asarray(p) - np.asarray(refl_pt)   ) +\
                                        np.linalg.norm( np.asarray(q) - np.asarray(refl_pt)  )

          
          ax.text(sites_sorted_by_x[tour_data.tour[-1]][0], sites_sorted_by_x[tour_data.tour[-1]][1], str(k),  
                  horizontalalignment='center', verticalalignment='center', fontsize=20, bbox=dict(facecolor='#ddcba0', alpha=1.0, pad=2.0))  
          

          # tour length for visiting sites in xorder
          x_order_tour_length    = 0.0
          base_order_tour_length = 0.0
          for (p,q) in zip(sites_sorted_by_x,sites_sorted_by_x[1:]):
                
                 refl_pt, refl_dist = calculate_reflection_point_and_distance(p,q)
                 x_order_tour_length += np.linalg.norm( np.asarray(p) - np.asarray(refl_pt)  ) +\
                                        np.linalg.norm( np.asarray(q) - np.asarray(refl_pt)  )

                 base_order_tour_length += p[1] + abs(q[0]-p[1]) + q[1]



          print Fore.RED, "Optimal order tour length for drone ", optimal_tour_length
          print Fore.RED, "X_order tour length"                 , x_order_tour_length
          print Fore.RED, "Base Order tour length"              , base_order_tour_length
          #utils_algo.print_list(sites_sorted_by_x)
          
          fig.canvas.draw()


def calculate_reflection_point_and_distance (  p, q ):
    """ For two points, p and q, calculate the reflection point 
    and distance between p and q
    """
    assert p[1] >= 0 and q[1] >= 0, "y cordinates of points should lie above the x-axis"   
    p = np.asarray(p)
    q = np.asarray(q)
    
    p_b = np.asarray([p[0],0])
    q_b = np.asarray([q[0],0])

    refl_pt    = (p[1]*q_b + q[1]*p_b)/(p[1]+q[1])
    refl_dist  = np.linalg.norm(p-refl_pt) + np.linalg.norm(q-refl_pt)
    return refl_pt, refl_dist
          

def applyAxCorrection(ax):
      ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
      ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
      ax.set_aspect(1.0)

def clearPatches(ax):
    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)


# Implementation of the key-press handler

def wrapperkeyPressHandler(fig,ax, run): 
      def _keyPressHandler(event):
                      
             if event.key in ['c', 'C']: 
                   # Clear canvas and states of all objects

                    ax.cla()
                    run.clearAllStates()


                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()

             elif event.key in ['a','A']: # run the greedy algorithm  to get a tour
                   run.getTour(fig,ax)

      return _keyPressHandler

# Implementation of \verb|wrapperEnterPoints|
   
def wrapperEnterPoints(fig,ax,run):
    def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert blue circle representing a site
                 newPoint = (event.xdata, event.ydata)
                 run.sites.append( newPoint  )
                 patchSize  = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/100.0

                 sitecolor = 'blue'

                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor=sitecolor, edgecolor='black'  ))
                 ax.set_title(r'Number of sites : ' + str(len(run.sites)), \
                              fontdict={'fontsize':20})

                 fig.canvas.draw()

                
    return _enterPoints


if __name__ == "__main__":
    # Body of main function
    
    fig, ax =  plt.subplots()
    run = SegmentHorsefly()
            
    ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
    ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
    ax.set_aspect(1.0)

    ax.set_xticks([])
    ax.set_yticks([])
              
    mouseClick = wrapperEnterPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
             
    keyPress   = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress )

    plt.show()
