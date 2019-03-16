
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
                            " (gncr)   Greedy NN Concentric Routing \n"                          +\
                            " (jte)    Greedy NN Joe thought experiments (phi < 1) \n"           +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'gncr':
                          tour = run.getTour( algo_greedy_nn_concentric_routing, phi)
                    elif algo_str == 'jte':
                          tour = run.getTour(algo_greedy_nn_joes_thought_experiments, phi)
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
def algo_greedy_nn_joes_thought_experiments(sites, inithorseposn, phi,    \
                                      write_algo_states_to_disk_p = True, \
                                      write_io_p                  = True, \
                                      animate_tour_p              = False,\
                                      plot_tour_p                 = True) :
    
    assert(phi<1) # If phi > 1 then you might have to do some waiting time. 
    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    from sklearn.neighbors import NearestNeighbors

    algo_name     = 'algo-greedy-nn-joes-thought-experiments'
    time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    dir_name      = algo_name + '---' + time_stamp
    io_file_name  = 'input_and_output.yml'

    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # First we decide the order in which the drones are collected/flies are serviced
    # i.e. we perform the thought experiments. While deciding the order. 
    # remember there is no waiting time. 
    numsites                 = len(sites)
    current_horse_posn       = np.asarray(inithorseposn) 
    horse_traj_seq           = [current_horse_posn]
    drones_collected_seq_p   = [False for i in range(numsites)] 
    order_collection         = []

    while not(all(drones_collected_seq_p)):
        uncollected_drones_seq_idx  = [idx for idx in range(len(drones_collected_seq_p)) 
                                      if drones_collected_seq_p[idx] == False]

        # find the closest uncollected drone from the current horse position
        # Replace this step with some dynamic nearest neighbor algorithm for 
        # improving the speed. 
        imin = 0 
        dmin = np.inf
        for idx in uncollected_drones_seq_idx:
            dmin_test  = np.linalg.norm(sites[idx]-current_horse_posn)
            if dmin_test < dmin:
                imin = idx
                dmin =  dmin_test

        # Make the horse and drone meet on a point along the 
        # segment joining the horse and drone
        new_horse_posn     = current_horse_posn + 1.0/(1+phi) * (sites[imin]-current_horse_posn) 
        current_horse_posn = new_horse_posn
        horse_traj_seq.append(current_horse_posn)
        order_collection.append((imin, new_horse_posn)) 

        # Mark the drone as collected and the position of the new horse
        drones_collected_seq_p[imin] = True

    # Plot the horse-trajectory and the corresponding collection points
    import matplotlib.animation as animation
    from   matplotlib.patches import Circle
    import matplotlib.pyplot as plt 

    # Set up configurations and parameters for all necessary graphics
       
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    sitesize = 0.015
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_aspect('equal')
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    ax[1].set_aspect('equal')

    ax[0].set_xticks(np.arange(0, 1, 0.1))
    ax[0].set_yticks(np.arange(0, 1, 0.1))
    ax[1].set_xticks(np.arange(0, 1, 0.1))
    ax[1].set_yticks(np.arange(0, 1, 0.1))

    # Turn on the minor TICKS, which are required for the minor GRID
    ax[0].minorticks_on()
    ax[1].minorticks_on()

    # customize the major grid
    ax[0].grid(which='major', linestyle='--', linewidth='0.3', color='red')
    ax[1].grid(which='major', linestyle='--', linewidth='0.3', color='red')

    # Customize the minor grid
    ax[0].grid(which='minor', linestyle=':', linewidth='0.3', color='black')
    ax[1].grid(which='minor', linestyle=':', linewidth='0.3', color='black')

    ax[0].get_xaxis().set_ticklabels([])
    ax[0].get_yaxis().set_ticklabels([])
    ax[1].get_xaxis().set_ticklabels([])
    ax[1].get_yaxis().set_ticklabels([])

    # Plot sites as small disks
    for site in sites:
         circle = Circle((site[0], site[1]), sitesize, \
                          facecolor = 'k'   , \
                          edgecolor = 'black'     , \
                          linewidth=1.0)
         sitepatch = ax[0].add_patch(circle)

    circle = Circle((inithorseposn[0], inithorseposn[1]), 0.02, facecolor = '#D13131', edgecolor='black', linewidth=1.0)
    ax[0].add_patch(circle)

    xhs = [x for (x,y) in horse_traj_seq]
    yhs = [y for (x,y) in horse_traj_seq]
    ax[0].plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

    # Fly segments
    for (idx, endpt) in order_collection:
        print idx, endpt
        xfs = [sites[idx][0], endpt[0]]
        yfs = [sites[idx][1], endpt[1]]
        ax[0].plot(xfs,yfs,'-',linewidth=2.0, markersize=3, alpha=0.7, color='k')

    tour_length = utils_algo.length_polygonal_chain(zip(xhs, yhs))
    ax[0].set_title("Number of sites: " + str(len(sites)) + "\n Serial Collection tour length " +\
                    str(round(tour_length,4)), fontsize=15)
    ax[0].set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=15)

    def normalize(vec):
        unit_vec =  1.0/np.linalg.norm(vec) * vec
        assert( abs(np.linalg.norm(unit_vec)-1.0) < 1e-8 )
        return unit_vec

    # Return the smaller angle between two vectors
    def cosine_angle_between_two_vectors(v1, v2):
        dot_product      = np.dot(v1,v2)
        norm_v1, norm_v2 = map(np.linalg.norm, [v1, v2])
        return dot_product/(norm_v1*norm_v2)


    # Now that we know the order of collection and the destination of each of the drones,
    # the drones start moving in tandem along those rays. In the mean time the horse 
    # collects the drones in order as given. For the algorithm to not crash it is 
    # important that phi < 1, which is why the assert statement was put at the beginning 
    # of this code.

    horse_traj                = [np.asarray(inithorseposn)]
    current_posns_of_drones   = map(np.asarray, [sites[idx] for (idx, _)   in order_collection]) # this keeps changing
    heading_vectors_of_drones = [normalize(head_pt - sites[idx]) for (idx, head_pt) in order_collection] # this stays constant throughout the loop! 

    for i in range(len(order_collection)):

        # Horse catches up with the drone 
        # Update the position of the horse *and* that drone.  
        # Record the time catch_time taken to catch up with that drone

        current_horse_posn = horse_traj[-1]
        drone_truck_vector = np.asarray(current_horse_posn) - current_posns_of_drones[i]                                    
        l                  = np.linalg.norm(drone_truck_vector)                                                             
        cos_theta          = cosine_angle_between_two_vectors(drone_truck_vector, heading_vectors_of_drones[i])            
        t                  = 1.0/(1.0-phi**2) * l * (-phi*cos_theta + np.sqrt(phi**2 * cos_theta**2 + (1.0-phi**2)))
        d                  = t * phi                                                                               
        new_drone_posn     = current_posns_of_drones[i] + d * heading_vectors_of_drones[i]                        
        new_horse_posn     = new_drone_posn # horse and drone have just met up

        current_posns_of_drones[i] = new_drone_posn
        horse_traj.append(new_horse_posn)

        # Update the positions of the remaining drones using catch_time recorded above
        # the current positions of the drones and the heading-vectors
        for k in range(i+1,len(current_posns_of_drones)):
                current_posns_of_drones[k] = current_posns_of_drones[k] +\
                                             d * heading_vectors_of_drones[k]
        
    # Initial position of horse
    circle = Circle((inithorseposn[0], inithorseposn[1]), 0.02, facecolor = '#D13131', edgecolor='black', linewidth=1.0)
    ax[1].add_patch(circle)

    # Horse tour
    xhs = [x for (x,y) in horse_traj]
    yhs = [y for (x,y) in horse_traj]
    ax[1].plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

    tour_length = utils_algo.length_polygonal_chain(zip(xhs, yhs))
    ax[1].set_title("Number of sites: " + str(len(sites)) + "\n Thought Experiment heuristic tour length " +\
                 str(round(tour_length,4)), fontsize=15)
    ax[1].set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=15)

    # Fly segments
    for (idx, endpt), i in zip(order_collection, range(len(current_posns_of_drones))):
        xfs = [sites[idx][0], endpt[0], current_posns_of_drones[i][0]]
        yfs = [sites[idx][1], endpt[1], current_posns_of_drones[i][1]]
        ax[1].plot(xfs,yfs,'s-',linewidth=2.0, markersize=5, markerfacecolor='g', alpha=0.7, color='k')

    # Plot sites as small disks
    for site in sites:
         circle = Circle((site[0], site[1]), sitesize, \
                          facecolor = 'k'   , \
                          edgecolor = 'black'     , \
                          linewidth=1.0)
         sitepatch = ax[1].add_patch(circle)



    plt.show()

    sys.exit()

   
def algo_greedy_nn_concentric_routing(sites, inithorseposn, phi, \
                                      write_algo_states_to_disk_p = True,\
                                      write_io_p                  = True,\
                                      animate_tour_p              = True,\
                                      plot_tour_p                 = False) :
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

    for idx in range(len(sites)):
         print "\n", Fore.GREEN, "Fly Trajectory ", idx, " is "
         utils_algo.print_list(fly_trajs[idx])
    print Style.RESET_ALL

    if plot_tour_p:
        plot_tour_gncr(sites       = sites, 
                  inithorseposn    = inithorseposn, 
                  phi              = phi, 
                  horse_trajectory = horse_traj, 
                  fly_trajectories = fly_trajs,
                  plot_file_name   = dir_name + '/' + 'plot.png')


    # Animate compute tour if \verb|animate_tour_p == True|
     
    if animate_tour_p:
        # strip away some of the waiting time gunk you had put in before.
        # Just pass in the points
        horse_traj = [elt[0] for elt in horse_traj]
        fly_trajs  = [[elt[0] for elt in traj] for traj in fly_trajs]
        animate_tour(phi                = phi, 
                     horse_trajectories = [horse_traj],
                     fly_trajectories   = fly_trajs,
                     animation_file_name_prefix = dir_name + '/' + io_file_name,
                     shortcut_fly_squiggles_p                = False)
    


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




def plot_tour_gncr (sites, inithorseposn, phi, \
               horse_trajectory,          \
               fly_trajectories,          \
               plot_file_name):

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

    horse_trajectory_pts = map(lambda x: x[0], horse_trajectory)
    tour_length = utils_algo.length_polygonal_chain(horse_trajectory_pts)
        
    ax.set_title("Number of sites: " + str(len(sites)) + "\nReverse Horsefly Tour Length: " +\
                 str(round(tour_length,4)), fontsize=15)
    ax.set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=15)

    xhs = [ elt[0][0] for elt in horse_trajectory] 
    yhs = [ elt[0][1] for elt in horse_trajectory] 
    ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

    circle = Circle((inithorseposn[0], inithorseposn[1]), 0.02, 
                    facecolor = 'r', edgecolor='black', linewidth=1.0)
    ax.add_patch(circle)
    
    # Plot sites as black circles
    for site in sites:
         circle = Circle((site[0], site[1]), 0.005, \
                          facecolor = 'k'   , \
                          edgecolor = 'black'     , \
                          linewidth=1.0)
         sitepatch = ax.add_patch(circle)

    # Plot the routes of flies as black lines
    for ftraj in fly_trajectories:
        xfs = [elt[0][0] for elt in ftraj ]
        yfs = [elt[0][1] for elt in ftraj ]
        
        #xfs = [ftraj[0][0][0], ftraj[-1][0][0]]
        #yfs = [ftraj[0][0][1], ftraj[-1][0][1]]
        
        ax.plot(xfs,yfs,'-',linewidth=2.0, markersize=6, alpha=0.5, color='k')

    plt.savefig(plot_file_name, bbox_inches='tight', dpi=250)
    plt.show()

#----------------------------------------------------------------------------------------
def animate_tour (phi, horse_trajectories, fly_trajectories, 
                  animation_file_name_prefix):
    """ This function can handle the animation of multiple
    horses and flies even when the the fly trajectories are all squiggly
    and if the flies have to wait at the end of their trajectories. 
    
    A fly trajectory should only be a list of points! The sites are always the 
    first points on the trajectories. Any waiting for the flies, is assumed to be 
    at the end of their trajectories where it waits for the horse to come 
    and pick them up. 

    Every point on the horse trajectory stores a list of indices of the flies
    collected at the end point. (The first point just stores the dummy value None). 
    Usually these index lists will be size 1, but there may be heuristics where you 
    might want to collect a bunch of them together since they may already be waiting 
    there at the pick up point. 

    For each drone collected, a yellow circle is placed on top of it, so that 
    it is marked as collected to be able to see the progress of the visualization 
    as it goes on. 
    """
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

    ############ Here be dragons....updating the length of the horse trajectory when multiple points involved and waiting of horses possibly?
    horse_trajectory_pts = map(lambda x: x[0], horse_trajectories[0])
    tour_length = utils_algo.length_polygonal_chain(horse_trajectory_pts)
    ax.set_title("Number of drones: " + str(len(fly_trajectories)) + "\nTour Length: " + str(round(tour_length,4)), fontsize=19)
    ax.set_xlabel(r"$\varphi=$ " + str(phi), fontsize=19)
    ###########3

    number_of_flies  = len(fly_trajectories)
    number_of_horses = len(horse_trajectories)
    colors           = utils_graphics.get_colors(number_of_horses, lightness=0.5)
        
    ims                = []
    
    # Constant for discretizing each segment inside the trajectories of the horses
    # and flies. 
    NUM_SUB_LEGS              = 3 # Number of subsegments within each segment of every trajectory
    
    # Arrays keeping track of the states of the horses
    horses_reached_endpt_p    = [False for i in range(number_of_horses)]
    horses_traj_num_legs      = [len(traj)-1 for traj in horse_trajectories] # the -1 is because the initial position of the horse is always included. 
    horses_current_leg_idx    = [0 for i in range(number_of_horses)]
    horses_current_subleg_idx = [0 for i in range(number_of_horses)] 
    horses_current_posn       = [traj[0] for traj in horse_trajectories]

    # Arrays keeping track of the states of the flies
    flies_reached_endpt_p    = [False for i in range(number_of_flies)]
    flies_traj_num_legs      = [len(traj)-1 for traj in fly_trajectories]
    flies_current_leg_idx    = [0 for i in range(number_of_flies)]
    flies_current_subleg_idx = [0 for i in range(number_of_flies)] 
    flies_current_posn       = [traj[0] for traj in fly_trajectories]

    # The drone collection process ends, when all the flies AND horses 
    # have reached their ends. Some heuristics, might involve the flies 
    # or the horses waiting at the endpoints of their respective trajectories. 
    image_frame_counter = 0
    while not(all(horses_reached_endpt_p + flies_reached_endpt_p)): 

        # Update the states of all the horses
        for hidx in range(number_of_horses):
            if horses_reached_endpt_p[hidx] == False:
                htraj  = horse_trajectories[hidx]

                if horses_current_subleg_idx[hidx] <= NUM_SUB_LEGS-2:

                    horses_current_subleg_idx[hidx] += 1     # subleg idx changes
                    legidx    = horses_current_leg_idx[hidx] # the legidx remains the same
                    
                    sublegidx = horses_current_subleg_idx[hidx] # shorthand for easier reference in the next two lines
                    xcurr = np.linspace( htraj[legidx][0], htraj[legidx+1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( htraj[legidx][1], htraj[legidx+1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    horses_current_posn[hidx]  = [xcurr, ycurr] 

                else:
                    horses_current_subleg_idx[hidx] = 0 # reset to 0
                    horses_current_leg_idx[hidx]   += 1 # you have passed onto the next leg
                    legidx    = horses_current_leg_idx[hidx]

                    xcurr, ycurr = htraj[legidx][0], htraj[legidx][1] # current position is the zeroth point on the next leg
                    horses_current_posn[hidx]  = [xcurr , ycurr] 

                    if horses_current_leg_idx[hidx] == horses_traj_num_legs[hidx]:
                        horses_reached_endpt_p[hidx] = True

        # Update the states of all the flies
        for fidx in range(number_of_flies):
            if flies_reached_endpt_p[fidx] == False:
                ftraj  = fly_trajectories[fidx]

                if flies_current_subleg_idx[fidx] <= NUM_SUB_LEGS-2:
                    
                    flies_current_subleg_idx[fidx] += 1
                    legidx    = flies_current_leg_idx[fidx]

                    sublegidx = flies_current_subleg_idx[fidx]
                    xcurr = np.linspace( ftraj[legidx][0], ftraj[legidx+1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( ftraj[legidx][1], ftraj[legidx+1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    flies_current_posn[fidx]  = [xcurr, ycurr] 

                else:
                    flies_current_subleg_idx[fidx] = 0 # reset to zero
                    flies_current_leg_idx[fidx]   += 1 # you have passed onto the next leg
                    legidx    = flies_current_leg_idx[fidx]

                    xcurr, ycurr = ftraj[legidx][0], ftraj[legidx][1] # current position is the zeroth point on the next leg
                    flies_current_posn[fidx]  = [xcurr , ycurr] 

                    if flies_current_leg_idx[fidx] == flies_traj_num_legs[fidx]:
                        flies_reached_endpt_p[fidx] = True

        objs = []
        # Render all the horse trajectories uptil this point in time. 
        for hidx in range(number_of_horses):
            traj               = horse_trajectories[hidx]
            current_horse_posn = horses_current_posn[hidx]
            
            if horses_current_leg_idx[hidx] != horses_traj_num_legs[hidx]: # the horse is still moving

                  xhs = [traj[k][0] for k in range(1+horses_current_leg_idx[hidx])] + [current_horse_posn[0]]
                  yhs = [traj[k][1] for k in range(1+horses_current_leg_idx[hidx])] + [current_horse_posn[1]]

            else: # The horse has stopped moving
                  xhs = [x for (x,y) in traj]
                  yhs = [y for (x,y) in traj]

            horseline, = ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')
            horseloc   = Circle((current_horse_posn[0], current_horse_posn[1]), 0.015, facecolor = '#D13131', edgecolor='k',  alpha=1.00)
            horsepatch = ax.add_patch(horseloc)
            objs.append(horseline)
            objs.append(horsepatch)


        # Render all fly trajectories uptil this point in time
        for fidx in range(number_of_flies):
            traj               = fly_trajectories[fidx]
            current_fly_posn   = flies_current_posn[fidx]
            
            if flies_current_leg_idx[fidx] != flies_traj_num_legs[fidx]: # the fly is still moving

                  xfs = [traj[k][0] for k in range(1+flies_current_leg_idx[fidx])] + [current_fly_posn[0]]
                  yfs = [traj[k][1] for k in range(1+flies_current_leg_idx[fidx])] + [current_fly_posn[1]]

            else: # The fly has stopped moving
                  xfs = [x for (x,y) in traj]
                  yfs = [y for (x,y) in traj]

            ##### Only use the currrent position in the fly trajectories               
            xfs = [current_fly_posn[0]]
            yfs = [current_fly_posn[1]]
            ###### Experimental.....

            fly_line_col = 'b'
            #flyline, = ax.plot(xfs,yfs,'-',linewidth=2.5, markersize=6, alpha=0.2, color=fly_line_col)
            #objs.append(flyline)

            ###### Only use when there is no waiting on the part of both the fly or the horse
            if flies_reached_endpt_p[fidx] == True:
                service_status_col = 'y'
            else:
                service_status_col = fly_line_col
            ######## Experimental......

            flyloc   = Circle((current_fly_posn[0], current_fly_posn[1]), 0.007, 
                              facecolor = service_status_col, edgecolor='k', alpha=1.00)
            flypatch = ax.add_patch(flyloc)
            objs.append(flypatch)
            
        print "........................"
        print "Appending to ims ", image_frame_counter
        ims.append(objs) 
        image_frame_counter += 1

    from colorama import Back 
    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=70, blit=True)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=150)
    #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)
    plt.show()
