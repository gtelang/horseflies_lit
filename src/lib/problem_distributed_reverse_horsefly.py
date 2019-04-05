
# Relevant imports for distributed reverse horsefly

from colorama import Fore, Style
from matplotlib import rc
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import inspect 
import itertools
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('seaborn-poster')
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time
import utils_algo
import utils_graphics


# Set up logging information relevant to this module
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def debug(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.debug('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))


def info(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.info('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))

xlim, ylim = [0,1], [0,1]

def applyAxCorrection(ax):
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
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




def run_handler():
    # Define key-press handler
       
    # The key-stack argument is mutable! I am using this hack to my advantage.
    def wrapperkeyPressHandler(fig,ax, run): 
           def _keyPressHandler(event):
               if event.key in ['i', 'I']:  
                    # Start entering input from the command-line
                    # Set speed of the flies at the sites
                   
                    phi_str = raw_input(Fore.YELLOW + "What should I set the speed of the flies to be?: " + Style.RESET_ALL)
                    phi = float(phi_str)

                    sensing_radius_str = raw_input(Fore.YELLOW + "What should I set the sensing radius of the truck and the flies to be?: " + Style.RESET_ALL)
                    sensing_radius = float(sensing_radius_str)
                    
                    # Select algorithm to execute

                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (avg-head) Average Heading Heuristic\n"                           +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'avg-head':
                          tour = run.getTour( algo_average_heading, phi, sensing_radius)
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    utils_graphics.applyAxCorrection(ax)
                    plot_tour(ax, tour)
                    fig.canvas.draw()
                    
                    
               elif event.key in ['n', 'N', 'u', 'U']: 
                    # Generate a bunch of uniform or non-uniform random points on the canvas
                    numpts = int(raw_input("\n" + Fore.YELLOW+\
                                           "How many points should I generate?: "+\
                                           Style.RESET_ALL)) 
                    run.clearAllStates()
                    ax.cla()
                                   
                    utils_graphics.applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                    
                    fig.texts = []
                                     
                    import scipy
                    if event.key in ['n', 'N']: 
                            run.sites = utils_algo.bunch_of_non_uniform_random_points(numpts)
                    else : 
                            run.sites = scipy.rand(numpts,2).tolist()

                    patchSize  = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/140.0

                    for site in run.sites:      
                        ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                     facecolor='blue',edgecolor='black' ))

                    ax.set_title('Points : ' + str(len(run.sites)), fontdict={'fontsize':40})
                    fig.canvas.draw()
                    
               elif event.key in ['c', 'C']: 
                    # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    utils_graphics.applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
                    
           return _keyPressHandler
    
    # Set up interactive canvas
    fig, ax =  plt.subplots()
    run = DistributedReverseHorseflyInput()
    #print run
        
    ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
    ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    mouseClick   = wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
    
def wrapperEnterRunPoints(fig, ax, run):
    def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert blue circle representing a site
                 newPoint = (event.xdata, event.ydata)
                 run.sites.append( newPoint  )
                 patchSize  = (xlim[1]-xlim[0])/140.0
                    
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='blue', edgecolor='black'  ))
                 ax.set_title('Points Inserted: ' + str(len(run.sites)), \
                              fontdict={'fontsize':40})
                 

             elif event.button == 3:  
                 # Insert big red circle representing initial position of horse and fly
                 newinithorseposn = (event.xdata, event.ydata)
                 run.inithorseposn = newinithorseposn  
                 patchSize         = (xlim[1]-xlim[0])/100.0

                 ax.add_patch( mpl.patches.Circle( newinithorseposn,radius = patchSize,
                                                   facecolor= '#D13131', edgecolor='black' ))
                 
                 print Fore.RED, "Initial positions of horse\n", 
                 utils_algo.print_list(run.inithorseposn)
                 print Style.RESET_ALL

             # Clear polygon patches and set up last minute \verb|ax| tweaks
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()
             

    return _enterPoints


# Local data-structures
class DistributedReverseHorseflyInput:
      def __init__(self, sites=[], inithorseposn=None):
           self.sites           = sites
           self.inithorseposn   = inithorseposn

      # Methods for \verb|DistributedReverseHorseflyInput|
      def clearAllStates (self):
          self.sites = []
          self.inithorseposn = None

      def getTour(self, algo, phi, sensing_radius):
                 return algo(self.sites, self.inithorseposn, phi, sensing_radius)

#-----------------------------------------------
# Helper functions
#------------------------------------------------
def get_random_direction():
    angle = np.random.uniform(0.0, 2.0*np.pi)
    return [np.cos(angle), np.sin(angle)]


def center_minimum_enclosing_rectangle(pts):
    
    assert(len(pts) > 0)

    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]


    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xc = (xmin+xmax)/2.0
    yc = (ymin+ymax)/2.0

    return np.asarray([xc, yc])

def outside_unit_bounding_box_p(pt):
    if pt[0] < 1.0 and pt[0] > 0.0 and pt[1] < 1.0 and pt[1] > 0.0:
       return True
    else :
       return False

def normalize(vec):
    assert(len(vec) == 2)
    vec = np.asarray(vec)
    return 1.0/np.linalg.norm(vec) * vec

#------------------------------------------------
# Algorithms for distributed reverse horsefly
#------------------------------------------------
def algo_average_heading(sites, inithorseposn, phi, sensing_radius, plot_tour_p=True):
    """ Each drone heads in the average direction of his $\delta$ neighbors
    except those that have the truck in $\delta$ neighborhood. They head 
    towards the truck. [if none, then random direction] 

    What does the truck do? It heads towards the center of mass of the 
    nearby drones in $\delta$ neighborhood. 

    Both the truck and the drones have the same sensing radius. So for a 
    given truck and drone either they both sense each other or they 
    both don't sense each other. 
    """
    import scipy.spatial as spatial

    # Set algo-state and input-output files config
    import sys, datetime, os, errno

    algo_name = 'algo-average-heading-distributed-reverse-horsefly'
    time_stamp= datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    dir_name  = algo_name + '---' + time_stamp

    try:
          os.makedirs(dir_name)
    except OSError as e:
          if e.errno != errno.EEXIST:
             raise
    



    numflies          = len(sites)
    horse_traj        = [ {'coords'              : np.asarray(inithorseposn), 
                            'fly_idxs_picked_up' : []                       , 
                            'global_clock_time'  : 0.0} ] 
    fly_trajs         = [[np.array(sites[i])] for i in range(numflies)] 
    flies_collected_p = [False for i in range(numflies)]

    old_drone_dirns   = [get_random_direction() for _ in range(numflies)]
    new_drone_dirns   = [np.asarray([0.0, 0.0]) for _ in range(numflies)]
    
    horse_dirn        = get_random_direction()
    horse_static_p    = False

    global_clock_time = 0.0
    dt                = 1e-2 # After each interval of time dt, each drone looks to 
                             # its neighbors and recomputes the heading direction
    tol               = 1e-2 # if the length of a vector is less than tol, it is considered as zero. 
 
    max_global_clock_time = 1.5
    plot_number = 0
   
    # Each iteration of the loop, updates the global clock by dt. 
    # The iteration ends when all drones are marked as collected. 
    # or when the global clock time exceeds a prespecified max-time.
    while ((not all(flies_collected_p))  and \
           (global_clock_time < max_global_clock_time)):

        global_clock_time += dt
        plot_number       += 1

        #-----------------------------------------------------------------------------
        # Update information about the horse and mark drones as collected if they come
        # sufficiently near the horse
        #-----------------------------------------------------------------------------
        current_horse_posn    = horse_traj[-1]['coords']
        unserviced_flies_idxs = [idx for idx in range(len(flies_collected_p)) 
                                  if flies_collected_p[idx] == False]

        current_fly_posns     = [fly_trajs[k][-1] for k in unserviced_flies_idxs]
        point_tree            = spatial.cKDTree(current_fly_posns) # for nearest neighbor range searches
        horse_delta_nbrs_idxs = point_tree.query_ball_point(current_horse_posn, sensing_radius) 
        
        # Case 1: Truck has *at least one* drone in its sensing neighborhood
        if horse_delta_nbrs_idxs:

            # Case 1A: Truck is currently static
            if horse_static_p: 
                horse_heading_pt = current_horse_posn 

                # Mark a drone as collected if it is sufficiently close to the truck
                marked_drone_idxs = []
                for idx in unserviced_flies_idxs:
                    current_idx_fly_posn =  fly_trajs[idx][-1]
                    if np.linalg.norm( current_idx_fly_posn - current_horse_posn ) < tol:
                        marked_drone_idxs.append(idx)

                # Marked drones are now deemed as collected
                for i in range(numflies):
                    if i in marked_drone_idxs:
                        flies_collected_p[i] = True
                
                horse_traj.append({'coords'             : current_horse_posn, 
                                   'fly_idxs_picked_up' : marked_drone_idxs,
                                   'global_clock_time'  : global_clock_time})

            # Case 1B: Truck is currently moving
            else:  
                horse_heading_pt   = center_minimum_enclosing_rectangle(\
                                         [current_fly_posns[idx] for idx in unserviced_flies_idxs] +\
                                         [current_horse_posn])
                tmp                = horse_heading_pt - current_horse_posn
                horse_dirn         = 1.0/np.linalg.norm(tmp) * tmp
                current_horse_posn = current_horse_posn + 1.0 * dt * horse_dirn 

                # Mark truck as static if it has reached its heading point
                if np.linalg.norm( current_horse_posn - horse_heading_pt ) < tol: 
                    current_horse_posn = horse_heading_pt
                    horse_static_p     = True

                horse_traj.append({'coords'             : current_horse_posn, 
                                   'fly_idxs_picked_up' : [],
                                   'global_clock_time'  : global_clock_time})
       
        # Case 2: Truck does *not* have *any* drones in its sensing neighborhood
        else: # choose a random direction to go in such that in the new position, you still remain inside the bounding box.
            horse_heading_pt = None
            horse_static_p   = False
            
            horse_dirn           = get_random_direction()
            tentative_horse_posn = current_horse_posn + 1.0 * dt * horse_dirn

            while outside_bounding_box_p(tentative_horse_posn):
                horse_dirn           = get_random_direction()
                tentative_horse_posn = current_horse_posn + 1.0 * dt * horse_dirn
            
            current_horse_posn = tentative_horse_posn 
            horse_traj.append({'coords'             : current_horse_posn, 
                               'fly_idxs_picked_up' : [],
                               'global_clock_time'  : global_clock_time})

        #------------------------------------------------------------------
        # Recalculate list of unserviced drones and update their positions
        #------------------------------------------------------------------
        unserviced_flies_idxs = [idx for idx in range(len(flies_collected_p)) 
                                 if flies_collected_p[idx] == False]
        current_fly_posns    = [fly_trajs[k][-1] for k in unserviced_flies_idxs]
        point_tree           = spatial.cKDTree(current_fly_posns) # for nearest neighbor range searches

        # Update positions of drones
        for idx in unserviced_flies_idxs:
            current_idx_fly_posn =  fly_trajs[idx][-1]

            # Case 1: Current drone and Truck *can* sense each other
            if np.linalg.norm(current_idx_fly_posn - current_horse_posn) < sensing_radius:

                new_dirn             = horse_heading_pt - current_idx_fly_posn
                new_drone_dirns[idx] = normalize(new_dirn)

            # Case 2: Current drone and truck *cannot* sense each other
            else: 
                # from https://stackoverflow.com/a/32424650
                fly_delta_nbrs_idxs = point_tree.query_ball_point(current_idx_fly_posn, sensing_radius) 
                
                if fly_delta_nbrs_idxs: 

                    sumvec = np.asarray([0.0, 0.0])
                    for jdx in fly_delta_nbrs_idxs:
                        sumvec += old_drone_dirns[jdx]
                    avgvec               = sumvec/len(fly_delta_nbrs_idxs)
                    new_drone_dirns[idx] = normalize(avgvec)

                else:
                    new_drone_dirns[idx] = get_random_direction()

            new_drone_posn = current_idx_fly_posn + phi * dt * new_drone_dirns[idx]
            fly_trajs[idx].append(new_drone_posn)

        print " Global clock time: "      , global_clock_time, "/", max_global_clock_time, \
              " Num flies collected: "    , sum(flies_collected_p), \
              " Current Horse Position: " , current_horse_posn

        if plot_tour_p:
            print Fore.GREEN, "..................................................................", Style.RESET_ALL
            plot_tour_so_far(horse_traj, fly_trajs, dir_name + '/' , plot_number, global_clock_time, sensing_radius)
            
        # https://stackoverflow.com/a/2612815 
        old_drone_dirns = list(new_drone_dirns) 

#----------------------------------------------------------------------------------------
# Render current positions of horse and flies to disk in the form of png files
# Stitch the images together with ffmpeg for an animation
# Make sure that the horse-tour so far is plotted, rendering fly trajectories
# will be optional. Collected drones will be marked with a different color
# Rendering fly trajectories in general can be very costly, so make sure you
# have only a few drones when working with that option. 
#----------------------------------------------------------------------------------------
def plot_tour_so_far(horse_traj, fly_trajs, file_prefix, plot_number, 
                     global_clock_time, sensing_radius):
    import numpy as np
    import matplotlib.animation as animation
    from   matplotlib.patches import Circle
    import matplotlib.pyplot as plt 

    # Set up configurations and parameters for all necessary graphics
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()

    # customize the major grid
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='red')

    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    # Indices of flies collected
    flies_idxs_collected = []
    for pt in horse_traj:
        flies_idxs_collected.extend(pt['fly_idxs_picked_up'])

    # Fly Trajectories
    for i in range(len(fly_trajs)):
        
         if i in flies_idxs_collected:
             facecolor = 'y'
         else :
             facecolor = 'b'

         # Location of the flies
         xloc = fly_trajs[i][-1][0]
         yloc = fly_trajs[i][-1][1]
         circle    = Circle((xloc, yloc), 0.010, \
                            facecolor=facecolor, edgecolor='black',linewidth=1.0)
         sitepatch = ax.add_patch(circle)

         # Sensing disks
         disk = Circle((xloc, yloc), sensing_radius, facecolor='y', 
                       edgecolor='black', linewidth=0.9, alpha=0.1)
         diskpatch = ax.add_patch(disk)


    # Horse Trajectory
    xhs = [ pt['coords'][0] for pt in horse_traj ]
    yhs = [ pt['coords'][1] for pt in horse_traj ]
    
    # Location of the horse
    horseloc   = Circle((xhs[-1], yhs[-1]), 0.015, facecolor = '#D13131', edgecolor='k',  alpha=1.00)
    horsepatch = ax.add_patch(horseloc)
    ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

    # Sensing disk of horse
    disk = Circle((xhs[-1], yhs[-1]), sensing_radius, facecolor='g', 
                  edgecolor='black', linewidth=1.0, alpha=0.2)
    diskpatch = ax.add_patch(disk)



    str_plot_number = format(plot_number, '05d')   
    plt.savefig(file_prefix + 'plot_' + str_plot_number, dpi=100, bbox_inches='tight')

