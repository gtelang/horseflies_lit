    
# Relevant imports for classic horsefly

from matplotlib import rc
from colorama import Fore
from colorama import Style
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time
import utils_algo
import utils_graphics

def run_handler():
    # Define key-press handler
       
    # The key-stack argument is mutable! I am using this hack to my advantage.
    def wrapperkeyPressHandler(fig,ax, run): 
           def _keyPressHandler(event):
               if event.key in ['i', 'I']:  
                    # Start entering input from the command-line
                    phi_str = raw_input(Fore.YELLOW + \
                              "Enter speed of fly (should be >1): " +\
                               Style.RESET_ALL)
                    phi = float(phi_str)

                    algo_str = raw_input(Fore.YELLOW + \
                              "Enter algorithm to be used to compute the tour:\n Options are:\n" +\
                            "  (e)   Exact \n"                                   +\
                            "  (t)   TSP   \n"                                   +\
                            "  (tl)  TSP   (using approximate L1 ordering)\n"    +\
                            "  (k)   k2-center   \n"                             +\
                            "  (kl)  k2-center (using approximate L1 ordering)\n"  +\
                            "  (g)   Greedy\n"                                   +\
                            "  (gl) Greedy (using approximate L1 ordering])  "  +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()

                    # Incase there are patches present from the previous clustering, just clear them
                    clearAxPolygonPatches(ax)

                    if   algo_str == 'e':
                          horseflytour = \
                                 run.getTour( algo_dumb,
                                              phi )
                    elif algo_str == 'k': 
                          horseflytour = \
                                 run.getTour( algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=algo_exact_given_specific_ordering)
                          print " "
                          print Fore.GREEN, answer['tour_points'], Style.RESET_ALL
                    elif algo_str == 'kl':
                          horseflytour = \
                                 run.getTour( algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=algo_approximate_L1_given_specific_ordering)
                    elif algo_str == 't':
                          horseflytour = \
                                 run.getTour( algo_tsp_ordering,
                                              phi,
                                              post_optimizer=algo_exact_given_specific_ordering)
                    elif algo_str == 'tl':
                          horseflytour = \
                                 run.getTour( algo_tsp_ordering,
                                              phi,
                                              post_optimizer= algo_approximate_L1_given_specific_ordering)
                    elif algo_str == 'g':
                          horseflytour = \
                                 run.getTour( algo_greedy,
                                              phi,
                                              post_optimizer= algo_exact_given_specific_ordering)
                    elif algo_str == 'gl':
                          horseflytour = \
                                 run.getTour( algo_greedy,
                                              phi,
                                              post_optimizer= algo_approximate_L1_given_specific_ordering)
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    #print horseflytour['tour_points']
                    plotTour(ax,horseflytour, run.inithorseposn, phi, algo_str)
                    applyAxCorrection(ax)
                    fig.canvas.draw()
                    
               elif event.key in ['n', 'N', 'u', 'U']: 
                    # Generate a bunch of uniform or non-uniform random points on the canvas
                    numpts = int(sys.argv[1]) 
                    run.clearAllStates()
                    ax.cla()
                                   
                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                    
                    fig.texts = []
                                     
                    import scipy
                    if event.key in ['n', 'N']: # Non-uniform random points
                            run.sites = utils_algo.bunch_of_random_points(numpts)
                    else : # Uniform random points
                            run.sites = scipy.rand(numpts,2).tolist()

                    patchSize  = (xlim[1]-xlim[0])/140.0

                    for site in run.sites:      
                        ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                     facecolor='blue',edgecolor='black' ))

                    ax.set_title('Points : ' + str(len(run.sites)), fontdict={'fontsize':40})
                    fig.canvas.draw()
                    
               elif event.key in ['c', 'C']: 
                    # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
                    
    
    # Set up interactive canvas
    fig, ax =  plt.subplots()
    run = HorseFlyInput()
    #print run
        
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    mouseClick   = wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
    

# Local data-structures for classic horsefly
class HorseFlyInput:
      def __init__(self, sites=[], inithorseposn=()):
           self.sites         = sites
           self.inithorseposn = inithorseposn

           
      def clearAllStates (self):
          """ Set the sites to an empty list and initial horse position 
          to the empty tuple.
          """
          self.sites = []
          self.inithorseposn = ()

          
      def getTour(self, algo, speedratio, k=None, post_optimizer=None):
          """ This method runs an appropriate algorithm for calculating
          a horsefly tour. The list of possible algorithms are 
          inside this module prefixed with 'algo_'
          
          The output is a dictionary of size 2, containing two lists,
          - Contains the vertices of the polygonal 
            path taken by the horse
          - The list of sites in the order 
            in which they are serviced by the tour, i.e. the order 
            in which the sites are serviced by the fly.
          """

          if k==None and post_optimizer==None:
                return algo(self.sites, self.inithorseposn, speedratio)
          elif k == None:
                return algo(self.sites, self.inithorseposn, speedratio, post_optimizer)
          else:
                #print Fore.RED, self.sites, Style.RESET_ALL
                return algo(self.sites, self.inithorseposn, speedratio, k, post_optimizer)
          
      def __repr__(self):
          """ Printed Representation of the Input for HorseFly
          """
          if self.sites != []:
              tmp = ''
              for site in self.sites:
                  tmp = tmp + '\n' + str(site)
              sites = "The list of sites to be serviced are " + tmp    
          else:
              sites = "The list of sites is empty"

          if self.inithorseposn != ():
              inithorseposn = "\nThe initial position of the horse is " + \
                               str(self.inithorseposn)
          else:
              inithorseposn = "\nThe initial position of the horse has not been specified"
              
          return sites + inithorseposn

# Local utility functions for classic horsefly
   
def tour_length(horseflyinit):
   def _tourlength (x):
         
        # the first point on the tour is the
        # initial position of horse and fly
        # Append this to the solution x = [x0,x1,x2,....]
        # at the front
        htour = np.append(horseflyinit, x)
        length = 0 

        for i in range(len(htour))[:-3:2]:
                length = length + \
                         np.linalg.norm([htour[i+2] - htour[i], \
                                         htour[i+3] - htour[i+1]]) 
        return length

   return _tourlength
   
def tour_length_with_waiting_time_included(tour_points, horse_waiting_times, horseflyinit):
      tour_points   = np.asarray([horseflyinit] + tour_points)
      tour_links    = zip(tour_points, tour_points[1:])

      # the +1 because the inital position has been tacked on at the beginning
      # the solvers written the tour points except for the starting position
      # because that is known and part of the input. For this function
      # I need to tack it on for tour length
      assert(len(tour_points) == len(horse_waiting_times)+1) 

      sum = 0
      for i in range(len(horse_waiting_times)):

          # Negative waiting times means drone/fly was waiting
          # at rendezvous point
          if horse_waiting_times[i] >= 0:
              wait = horse_waiting_times[i]
          else:
              wait = 0
              
          sum += wait + np.linalg.norm(tour_links[i][0] - tour_links[i][1], ord=2) # 
      return sum

# Algorithms for classic horsefly
   
def algo_greedy(sites, inithorseposn, phi, post_optimizer):
      """
      This implements the greedy algorithm for the canonical greedy
      algorithm for collinear horsefly, and then uses the ordering 
      obtained to get the exact tour for that given ordering.
      
      Many variations on this are possible. However, this algorithm
      is simple and may be more amenable to theoretical analysis. 
      
      We will need an inequality for collapsing chains however. 
      """
      def next_rendezvous_point_for_horse_and_fly(horseposn, site):
            """
            Just use the exact solution when there is a single site. 
            No need to use the collinear horse formula which you can 
            explicitly derive. That formula is  an important super-special 
            case however to benchmark quality of solution. 
            """

            horseflytour = algo_exact_given_specific_ordering([site], horseposn, phi)
            return horseflytour['tour_points'][-1]
      
      # Begin the recursion process where for a given initial
      # position of horse and fly and a given collection of sites
      # you find the nearst neighbor proceed according to segment
      # horsefly formula for just and one site, and for the new
      # position repeat the process for the remaining list of sites. 
      # The greedy approach can be extended to by finding the k
      # nearest neighbors, constructing the exact horsefly tour
      # there, at the exit point, you repeat by taking k nearest
      # neighbors and so on. 
      def greedy(current_horse_posn, remaining_sites):
            if len(remaining_sites) == 1:
                  return remaining_sites
            else:
                  # For reference see this link on how nn queries are performed. 
                  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html
                  # Warning this is inefficient!!! I am rebuilding the kd-tree at each step. 
                  # Right now, I am only doing this for convenience.
                  from scipy import spatial
                  tree = spatial.KDTree(remaining_sites)

                  # The next site to get serviced by the drone and horse
                  # is the one which is closest to the current position of the
                  # horse. 
                  pts           = np.array([current_horse_posn])
                  query_result  = tree.query(pts)
                  next_site_idx = query_result[1][0]
                  next_site     = remaining_sites[next_site_idx]

                  next_horse_posn = \
                        next_rendezvous_point_for_horse_and_fly(current_horse_posn, next_site)
                  #print remaining_sites
                  remaining_sites.pop(next_site_idx) # the pop method modifies the list in place. 
                  
                  return [ next_site ] + greedy (current_horse_posn = next_horse_posn, \
                                                 remaining_sites    = remaining_sites)

      sites1 = sites[:]
      sites_ordered_by_greedy = greedy(inithorseposn, remaining_sites=sites1)

      # Use exact solver for the post optimizer step
      answer = post_optimizer(sites_ordered_by_greedy, inithorseposn, phi)
      return answer

# Plotting routines for classic horsefly
 
def plotTour(ax,horseflytour, horseflyinit, phi, algo_str, tour_color='#d13131'):
    """ Plot the tour on the given canvas area
    """
   
    # Route for the horse
    xhs, yhs = [horseflyinit[0]], [horseflyinit[1]]
    for pt in horseflytour['tour_points']:
        xhs.append(pt[0])
        yhs.append(pt[1])

    # List of sites
    xsites, ysites = [], []
    for pt in horseflytour['site_ordering']:
        xsites.append(pt[0])
        ysites.append(pt[1])

    # Route for the fly. The fly keeps alternating
    # between the site and the horse
    xfs , yfs = [xhs[0]], [yhs[0]]
    for site, pt in zip (horseflytour['site_ordering'],
                         horseflytour['tour_points']):
        xfs.extend([site[0], pt[0]])
        yfs.extend([site[1], pt[1]])

    print "\n----------"
    print "Horse Tour"
    print "-----------"
    waiting_times = [0.0] + horseflytour['horse_waiting_times'].tolist() # the waiting time at the starting point is 0
    #print waiting_times
    for pt, time in zip(zip(xhs,yhs), waiting_times) :
        print pt, Fore.GREEN, " ---> Horse Waited ", time, Style.RESET_ALL

    print "\n----------"
    print "Fly Tour"
    print "----------"
    for item, i in zip(zip(xfs,yfs), range(len(xfs))):
        if i%2 == 0:
           print item
        else :
           print Fore.RED + str(item) + "----> Site" +  Style.RESET_ALL

    print "----------------------------------"
    print Fore.GREEN, "\nSpeed of the drone was set to be", phi
    #tour_length = utils_algo.length_polygonal_chain( zip(xhs, yhs))
    tour_length = horseflytour['tour_length_with_waiting_time_included']
    print "Tour length of the horse is ",  tour_length
    print "Algorithm code-Key used "    , algo_str, Style.RESET_ALL
    print "----------------------------------\n"
           
    #kwargs = {'size':'large'}
    for x,y,i in zip(xsites, ysites, range(len(xsites))):
          ax.text(x, y, str(i+1), bbox=dict(facecolor='#ddcba0', alpha=1.0)) 
    ax.plot(xfs,yfs,'g-') # fly tour is green
    ax.plot(xhs, yhs, color=tour_color, marker='s', linewidth=3.0) # horse is red


    # Initial position of horse and fly
    ax.add_patch( mpl.patches.Circle( horseflyinit,
                                      radius = 1/34.0,
                                      facecolor= '#D13131', #'red',
                                      edgecolor='black'   )  )


    fontsize = 10
    tnrfont = {'fontname':'Times New Roman'}
    ax.set_title(  'Algorithm Used: ' + algo_str +  '\nTour Length: ' \
                    + str(tour_length)[:7], fontdict={'fontsize':fontsize}, **tnrfont)
    ax.set_xlabel('Number of sites: ' + str(len(xsites)) + '\nDrone Speed: ' + str(phi) ,
                  fontdict={'fontsize':fontsize}, **tnrfont)

