
# Relevant imports for reverse horsefly

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


def run_handler():
    # Define key-press handler
       
    # The key-stack argument is mutable! I am using this hack to my advantage.
    def wrapperkeyPressHandler(fig,ax, run): 
           def _keyPressHandler(event):
               if event.key in ['i', 'I']:  
                    # Start entering input from the command-line
                    # Set speed of the flies at the sites
                    
                    phi_str = raw_input(Fore.YELLOW + "What should I set the speed of each of the flies at the sites to be? : " + Style.RESET_ALL)
                    phi = float(phi_str)
                    
                    # Select algorithm to execute

                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (gncr)   Greedy NN Concentric Routing \n"                                        +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'gncr':
                          tour = run.getTour( algo_greedy_nn_concentric_routing, phi)
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
    run = ReverseHorseflyInput()
    #print run
        
    ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
    ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    mouseClick   = utils_graphics.wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
    


# Local data-structures
class ReverseHorseflyInput:
      def __init__(self, sites=[], inithorseposn=()):
           self.sites           = sites
           self.inithorseposn   = inithorseposn

      # Methods for \verb|ReverseHorseflyInput|
      def clearAllStates (self):
         self.sites = []
         self.inithorseposn = ()

      def getTour(self, algo, speedratio):
            return algo(self.sites, self.inithorseposn, speedratio)

#---------------------------------
# Algorithms for reverse horsefly 
#---------------------------------   
def algo_greedy_nn_concentric_routing(sites, inithorseposn, phi, \
                                      write_algo_states_to_disk_p = True,\
                                      write_io_p                  = True,\
                                      animate_tour_p              = True) :
    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    from sklearn.neighbors import NearestNeighbors

    algo_name     = 'algo-greedy_nn_concentric_routing'
    time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    dir_name      = algo_name + '---' + time_stamp
    io_file_name  = 'input_and_output.yml'

    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    numsites            = len(sites)
    horse_traj          = [(np.asarray(inithorseposn), None)] 
    fly_trajs           = [[(np.array(sites[i]),None)] for i in range(numsites)] 
    packages_received_p = [False for i in range(numsites)]

    # There are exactly $n$ iterations in this loop where exactly 
    # one package gets exchanged between the horse and one of the 
    # flies. 
    while not(all(packages_received_p)):
        # CHOOSE CENTER: here we choose as 
        current_horse_posn   = horse_traj[-1][0]
        unserviced_flies_idx = [idx for idx in range(len(packages_received_p)) 
                                 if packages_received_p[idx] == False ]
        fly_trajs_unserviced = [fly_trajs[idx] for idx in unserviced_flies_idx]

        imin = 0
        dmin = np.inf
        for idx in unserviced_flies_idx:
            current_fly_posn = fly_trajs[idx][-1][0]
            dmin_test = np.linalg.norm(current_fly_posn-current_horse_posn)
            if dmin_test < dmin:
                imin = idx
                dmin =  dmin_test

        # the meeting point of the horse and the fly closest to the horse
        horse_nearest_fly_distance     = np.linalg.norm(current_horse_posn-fly_trajs[imin][-1][0])
        horse_fly_uvec                 = 1.0/horse_nearest_fly_distance * (fly_trajs[imin][-1][0]-current_horse_posn) 
        dist_of_horse_to_meeting_point = 1.0/(phi+1.0) * horse_nearest_fly_distance
        center_heading                 = current_horse_posn + dist_of_horse_to_meeting_point * horse_fly_uvec
       
        # Move the horse and flies
        new_horse_posn, \
        new_fly_posns , \
        subidx        = \
                    head_towards_center(center_heading         = center_heading, 
                                        current_horse_posn     = current_horse_posn, 
                                        current_posns_of_flies = [traj[-1][0] for traj in fly_trajs_unserviced], 
                                        phis                   = [phi for i in range(len(fly_trajs_unserviced))]) 

        # TACK ON NEW HORSE POSITION TO HORSE TRAJECTORY
        horse_traj.append(new_horse_posn) # note that the position also contains the waiting time for 
        print Fore.YELLOW, new_horse_posn, Style.RESET_ALL

        # The horse if any at the new position. In the greedy routing scheme this is always zero because 
        # you are heading towards the fly it can reach the soonest. Other schemes may choose different centers. 
        # TACK ON NEW FLY POSITION TO FLY TRAJECTORIES
        assert(len(unserviced_flies_idx) == len(new_fly_posns))
        for idx, posn in zip(unserviced_flies_idx, new_fly_posns):
            fly_trajs[idx].append(posn)    ##### Not all fly trajectories have to be updated

        # UPDATE packages_received_p,, flip the boolean flag on 
        # on the fly which just received its package. 
        packages_received_p[unserviced_flies_idx[subidx]] = True
 
    print "Yay! Reverse horsefly routing completed!"
    utils_algo.print_list(fly_trajs)
    print Fore.YELLOW, "Horse Trajectory is "
    utils_algo.print_list(horse_traj)
    print Style.RESET_ALL

    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(sites            = sites, 
                     inithorseposn    = inithorseposn, 
                     phi              = phi, 
                     horse_trajectory = horse_traj, 
                     fly_trajectories = fly_trajs,
                     animation_file_name_prefix = dir_name + '/' + io_file_name)
    


# This routine is the heart of all the concentric routing
# heuristics which is why I am trying to make it as general 
# as possible with varying fly speeds at the sites. The horse
# is assumed to have speed 1 though. 
# It returns 
# 1. The new positions of the horse and flies
# 2. Waiting times at the resulting points. Usually these waiting 
#    times are zero, except for those which reach the center
#    and has to wait either for the horse to get there, or if 
#    the horse got there then for the first fly to get there. Note that
#    it is possible for multiple flies to reach the center and 
#    having to wait. In this case, the horse hands over the packages
#    to the one with the least index, even if multiple flies have reached
#    the concentric point. It is this index which is returned by the function
#    along with the rendezvous points with waiting times. 
# What this function returns is important to the flow of the actual algorithm
# function that essentially only otherwise makes the decisions of what centers
# to choose to head towards. 
# Finally the money-shot begins. Implement this function carefully after some 
# detailed derivation. The important thing here is if horse reaches first, then 
# it waits and if the fly reaches first then it waits, basically you need to 
# calculate the time it takes for the horse to reach and the time it takes for 
# the fly to reach. Any waiting is the absolute value of the difference between 
# those two times. You just need to add it to the appropriate agent, and return 
# the appropriate index of the fly. 
def head_towards_center(center_heading,        \
                        current_horse_posn,    \
                        current_posns_of_flies,\
                        phis):

    assert(len(phis) == len(current_posns_of_flies))
    fly_times_to_center_with_idx = \
              zip([np.linalg.norm(current_posns_of_flies[idx]-center_heading)/phis[idx] \
                          for idx in range(len(current_posns_of_flies))], 
                  range(len(current_posns_of_flies)))

    fly_times_to_center_with_idx_sorted = sorted(fly_times_to_center_with_idx, key = lambda tup: tup[0])
    fastest_fly_time_to_center          = fly_times_to_center_with_idx_sorted[0][0]
    idx_of_fastest_fly_to_center        = fly_times_to_center_with_idx_sorted[0][1] # this fly gets the package
    horse_time_to_center                = np.linalg.norm(current_horse_posn-center_heading)/1.0

    # We want to find the indices of the flies which reach the center faster
    # than the horse. if imin = -1 after the loop below, it means the horsr
    # reaches the fastest
    imin = -1
    for idx in range(len(fly_times_to_center_with_idx_sorted)):
        if fly_times_to_center_with_idx_sorted[idx][0] < horse_time_to_center:
            imin = idx
            continue
        else:
            break

    new_horse_posn = [center_heading, abs(horse_time_to_center-fastest_fly_time_to_center)]
    new_fly_posns  = []
    for idx in range(len(current_posns_of_flies)):
           v      = center_heading - current_posns_of_flies[idx]
           unit_v = 1.0/np.linalg.norm(v) * v
           new_fly_posns.append([ current_posns_of_flies[idx] + \
                                  phis[idx]*fastest_fly_time_to_center * unit_v, 0.0 ])

    return new_horse_posn, new_fly_posns, idx_of_fastest_fly_to_center


# Animation routines
# Make sure that the fly_tours are all light blue
# but the dots are black as they move. When a site 
# gets serviced it turns to blue. Unserviced sites
# remain black till they get serviced. The horse 
# trajectory as before is always red. When you graduate
# to two horses, then you can make the things 
# multicolored.

# For nn routing, where all the flies have the same velocity
# you can completely ignore waiting times, because no one ever
# waits in the nn routing ever. 
def animate_tour (sites, inithorseposn, phi, \
                  horse_trajectory,          \
                  fly_trajectories,          \
                  animation_file_name_prefix):
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

    # Visually distinct colors for displaying each flys trajectory in a different color 
    number_of_flies = len(fly_trajectories)

    horse_trajectory_pts = map(lambda x: x[0], horse_trajectory)
    tour_length = utils_algo.length_polygonal_chain(horse_trajectory_pts)
        
    ax.set_title("Number of sites: " + str(len(sites)) + "\nReverse Horsefly Tour Length: " +\
                 str(round(tour_length,4)), fontsize=15)
    ax.set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=15)

    # Leg list form for all horse trajectories
    horse_traj_ll = []
    for i in range(len(horse_trajectory)-1):
        horse_traj_ll.append((horse_trajectory[i][0], horse_trajectory[i+1][0]))

    # Leg list form for all fly trajectories
    fly_trajs_ll = []
    for fly_traj in fly_trajectories:
        fly_traj_ll = []
        for i in range(len(fly_traj)-1):
             fly_traj_ll.append((fly_traj[i][0], fly_traj[i+1][0]))
        fly_trajs_ll.append(fly_traj_ll)

    def discretize_leg(pts):
        subleg_pts = []
        if pts == None:
             return None
        else:
             k  = 6
             pts = map(np.asarray, pts)
             for p,q in zip(pts, pts[1:]):
                 tmp = []
                 for t in np.linspace(0,1,k): 
                       tmp.append((1-t)*p + t*q) 
                 subleg_pts.extend(tmp[:-1])

             subleg_pts.append(pts[-1])
             return subleg_pts

    ims                 = []
    horse_points_so_far = []
    fly_points_so_far   = [[] for i in range(len(sites))] 
    for idx in range(len(horse_traj_ll)):
        horse_points_so_far.append(horse_traj_ll[idx][0])       
        
        for k in range(len(sites)):
              if idx < len(fly_trajs_ll[k]):
                     fly_points_so_far[k].append(fly_trajs_ll[k][idx][0])
              else :
                     fly_points_so_far[k].append(fly_trajectories[k][-1][0])

        horse_leg_pts  = [horse_traj_ll[idx][0], horse_traj_ll[idx][1]]
        horse_leg_disc = discretize_leg(horse_leg_pts)   # list of points 
        print horse_leg_disc

        for k in range(len(horse_leg_disc)):
            current_horse_posn = horse_leg_disc[k]
            objs = []

            # Plot trajectory of horse
            xhs = [pt[0] for pt in horse_points_so_far] + [current_horse_posn[0]]
            yhs = [pt[1] for pt in horse_points_so_far] + [current_horse_posn[1]]
            horseline, = ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')
            horseloc   = Circle((current_horse_posn[0], current_horse_posn[1]), 0.02, facecolor = '#D13131', alpha=1.00)
            horsepatch = ax.add_patch(horseloc)
            objs.append(horseline)
            objs.append(horsepatch)

            # Plot sites as black circles
            for site in sites:
                 circle = Circle((site[0], site[1]), 0.01, \
                                  facecolor = 'k'   , \
                                  edgecolor = 'black'     , \
                                  linewidth=1.0)
                 sitepatch = ax.add_patch(circle)
                 objs.append(sitepatch)

            debug(Fore.CYAN + "Appending to ims "+ Style.RESET_ALL)
            ims.append(objs) 

    # Write animation of tour to disk and display in live window
    from colorama import Back 
    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=150)
    #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

    plt.show() # For displaying the animation in a live window. 
