
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
                    
                    phi_str = raw_input(Fore.YELLOW + "What should I set the speed of each of the flies at the sites to be? : " + Style.RESET_ALL)
                    phi = float(phi_str)
                    
                    # Select algorithm to execute

                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (gncr)   Greedy NN Concentric Routing \n"                          +\
                            " (gdr)    Greedy Dead reckoning                       \n"           +\
                            " (gkin)   Greedy Kinetic TSP towards center           \n"           +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'gncr':
                          tour = run.getTour( algo_greedy_nn_concentric_routing, phi)
                    elif algo_str == 'gdr':
                          tour = run.getTour(algo_greedy_dead_reckoning , phi)
                    elif algo_str == 'gkin':
                          tour = run.getTour(algo_greedy_concentric_kinetic_tsp, phi)
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
                 run.inithorseposns.append(newinithorseposn)  
                 patchSize         = (xlim[1]-xlim[0])/100.0

                 ax.add_patch( mpl.patches.Circle( newinithorseposn,radius = patchSize,
                                                   facecolor= '#D13131', edgecolor='black' ))
                 
                 print Fore.RED, "Initial positions of horses\n", 
                 utils_algo.print_list(run.inithorseposns)
                 print Style.RESET_ALL

             # Clear polygon patches and set up last minute \verb|ax| tweaks
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()
             

    return _enterPoints


# Local data-structures
class ReverseHorseflyInput:
      def __init__(self, sites=[], inithorseposns=[]):
           self.sites            = sites
           self.inithorseposns   = inithorseposns

      # Methods for \verb|ReverseHorseflyInput|
      def clearAllStates (self):
          self.sites = []
          self.inithorseposns = []

      def getTour(self, algo, speedratio):
          if len(self.inithorseposns) == 1 and (id(algo) in map(id, [algo_greedy_concentric_kinetic_tsp, 
                                                                     algo_greedy_nn_concentric_routing, 
                                                                     algo_greedy_dead_reckoning]) ): # these algorithms can only handle one horse
                 return algo(self.sites, self.inithorseposns[0], speedratio)
          else:
                 return algo(self.sites, self.inithorseposns, speedratio)

#---------------------------------
# Algorithms for reverse horsefly 
#---------------------------------
def algo_greedy_dead_reckoning(sites, inithorseposn, phi,    \
                               write_algo_states_to_disk_p = False, \
                               write_io_p                  = False, \
                               animate_tour_p              = True,\
                               plot_tour_p                 = False) :
    
    # Set algo-state and input-output files config
    import sys, datetime, os, errno

    if write_io_p:
        algo_name     = 'algo-greedy-dead-reckoning'
        time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
        dir_name      = algo_name + '---' + time_stamp
        io_file_name  = 'input_and_output.yml'

        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    numsites           = len(sites)
    numflies           = numsites
    current_horse_posn = np.asarray(inithorseposn) 
    # horse_traj         = [current_horse_posn]

    horse_traj = [ {'coords'            : np.asarray(inithorseposn), 
                    'fly_idxs_picked_up': []                       , 
                    'waiting_time'      : 0.0} ] 

    flies_collected_p  = [False for i in range(numsites)] 
    order_collection   = []
    clock_time         = 0.0


    def normalize(vec):
        unit_vec =  1.0/np.linalg.norm(vec) * vec
        assert( abs(np.linalg.norm(unit_vec)-1.0) < 1e-8 ) # make sure vector has been properly normalized
        return unit_vec

    def nearest_uncollected_fly(current_horse_posn, uncollected_flies_idx):
        """ Find the closest uncollected fly from the current horse position
        Replace this step with some dynamic nearest neighbor algorithm for 
        improving the speed."""
        imin = 0 
        dmin = np.inf
        for idx in uncollected_flies_idx:
            dmin_test  = np.linalg.norm(sites[idx]-current_horse_posn)
            if dmin_test < dmin:
                imin = idx
                dmin = dmin_test
        return imin, dmin

    # Main loop
    while (not all(flies_collected_p)):
        current_horse_posn     = horse_traj[-1]['coords']
        uncollected_flies_idx  = [idx for idx in range(len(flies_collected_p)) if flies_collected_p[idx] == False]
        imin, dmin             = nearest_uncollected_fly(current_horse_posn, uncollected_flies_idx)

        fly_posn_dr    = sites[imin] + clock_time * phi * normalize(current_horse_posn-sites[imin]) # fly position just before dead reckoning begins
        heading_vector = normalize(fly_posn_dr - current_horse_posn)       # unit-vector of the direction in which the horse now heads
        d              = np.linalg.norm(fly_posn_dr - current_horse_posn)  # distance between the current position of the fly and the horse 
        new_horse_posn = current_horse_posn + d/(1.0+phi) * heading_vector # the meeting point of the horse and fly

        horse_traj.append({'coords'             : new_horse_posn, 
                           'fly_idxs_picked_up' : [imin]        , 
                           'waiting_time'       : 0.0})

        # truck speed is 1.0 and horse never waits, 
        # so clock_time = distance travelled by horse
        horse_traj_pts          = [pt['coords'] for pt in horse_traj ]
        clock_time              = utils_algo.length_polygonal_chain(horse_traj_pts) 
        flies_collected_p[imin] = True

        order_collection.append((imin, new_horse_posn, clock_time)) 

    # For the purposes of the animation we need to record where *each* fly 
    # is when some fly is collected. This step needs to be done outside 
    # the main loop above because before that loop I don't know where 
    # the flies are headed
    fly_trajs         = [[np.array(sites[i])] for i in range(numflies)]
    flies_collected_p = [False for i in range(numflies)]
    old_clock_time    = 0.0
    

    for (_, _, new_clock_time), idx in zip(order_collection, range(len(order_collection))):
        
        # Time difference by which all trajectories need to be updated
        dt = new_clock_time - old_clock_time

        # Update the trajectories of the uncollected flies beginning with the current fly
        for k, intercept_pt, _ in order_collection[idx:] : 

            current_fly_posn = fly_trajs[k][-1]
            heading_vector   = normalize(intercept_pt - current_fly_posn)
            newpt            = current_fly_posn + dt * phi * heading_vector
            fly_trajs[k].append(newpt)
   
        old_clock_time = new_clock_time
    
    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(phi                = phi, 
                     horse_trajectories = [horse_traj],
                     fly_trajectories   = fly_trajs,
                     animation_file_name_prefix = None,
                     algo_name = 'gdr', 
                     render_trajectory_trails_p = True)

    if plot_tour_p:
        from   matplotlib.patches import Circle
        import matplotlib.pyplot as plt 

        # Set up configurations and parameters for all necessary graphics
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        sitesize = 0.010
        fig, ax = plt.subplots()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_aspect('equal')

        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_yticks(np.arange(0, 1, 0.1))

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='--', linewidth='0.3', color='red')

        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        # Plot initial position of the horse 
        circle = Circle((inithorseposn[0], inithorseposn[1]), 0.02, facecolor = '#D13131', edgecolor='black', linewidth=1.0)
        ax.add_patch(circle)
        
        # Plot Horse tour
        xhs = [ pt['coords'][0] for pt in horse_traj ]
        yhs = [ pt['coords'][1] for pt in horse_traj ]
        ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

        # Plot sites as small disks (these are obviosuly the initial positions of the flies)
        for site in sites:
            circle    = Circle((site[0], site[1]), sitesize, facecolor='k', edgecolor='black',linewidth=1.0)
            sitepatch = ax.add_patch(circle)

        # Plot Fly segments
        for idx, endpt, _  in order_collection:
            print idx, endpt
            xfs = [sites[idx][0], endpt[0]]
            yfs = [sites[idx][1], endpt[1]]
            ax.plot(xfs,yfs,'-',linewidth=2.0, markersize=3, alpha=0.7, color='k')

        # Plot meta-data
        tour_length = utils_algo.length_polygonal_chain(zip(xhs, yhs))
        ax.set_title("Number of sites: " + str(len(sites)) + "\n Tour length " +\
                    str(round(tour_length,4)), fontsize=15)
        ax.set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=15)
        plt.show()

    return horse_traj, fly_trajs

#----------------------------------------------------------------------------------------------------------------------
def algo_greedy_concentric_kinetic_tsp(sites, inithorseposn, phi, \
                                       center_choice               = 'gncr_endpt',
                                       write_algo_states_to_disk_p = True,\
                                       write_io_p                  = True,\
                                       animate_tour_p              = True,\
                                       plot_tour_p                 = False) :

    """Each fly heads towards a given center-point (same center-point for all flies) 
    along the directed segment joining its initial position to the center. 
    If the fly reaches the center before it has been collected by the horse 
    it just waits there. 

    The horse in the mean-time executes a greedy-kinetic tsp, on the flies. This means 
    it just heads towards the nearest drone it can intercept. Note that the drone is moving 
    obloviously of the horse and does *not* dead-reckon for the horse the moment it knows
    it is about to be interecepted. 
    
    The default choice of center is the one generated by the greedy concentric routing 
    heuristic. Other center-choices are possible too, depending on the string passed 
    an appropriate computation of the center will be made. 

    My hope is that OPT for reverse horsefly can be deformed by a constant to an 
    OPT of this type. Placing an appropriate center to move towards along with an 
    initial bus depot position are interesting problems to consider. This would be a
    good result to have. 
    
    The advantage of solving this special type of reverse horsefly is that drones
    can be collected quickly and a single message can be broadcast to all drones 
    telling head towards here. The message complexity is reduced. 
    """
      
    # In this choice, all drones head towards the last point on the horse 
    # trajectory of the output of gncr
    if center_choice == 'gncr_endpt':     
        horse_traj_gncr, _ = algo_greedy_nn_concentric_routing(sites, inithorseposn, phi, 
                              write_algo_states_to_disk_p = False, write_io_p = False,
                              plot_tour_p = False, animate_tour_p = False)
        center = horse_traj_gncr[-1]['coords']

    # Else all drones head towards the initial position of horse
    elif center_choice == 'inithorseposn': 
         center = inithorseposn
         
    # Else all drones head towards the center of the minimum 
    # enclosing ball of the inithorseposn and the sites
    elif center_choice == 'meb_center':   
          pass

    else:                      
        print "Option not recognized. Please provide a center for all the flies to head towards"
        sys.exit()


    numflies             = len(sites)
    horse_traj           = [ {'coords'            : np.asarray(inithorseposn), 
                             'fly_idxs_picked_up': []                       , 
                             'waiting_time'      : 0.0} ] 
    fly_trajs              = [[np.array(sites[i])] for i in range(numflies)] 
    flies_collected_p      =  [False for i in range(numflies)]
    flies_reached_center_p =  [False for i in range(numflies)] 


    #print "\n----------------\nCenter used is ", center, "\n--------------------"

    # There are exactly $n$ iterations in this loop where exactly 
    # one fly is picked up by the horse per iteration. 
    #print flies_collected_p
    #print "Length of flies_collected_p ", len(flies_collected_p)

    while not(all(flies_collected_p)):

        current_horse_posn    = horse_traj[-1]['coords']

        # Find the index of the fly that can be intercepted at the 
        # earliest by the horse and the time to interception. 
        # While calculating this take into consideration whether the 
        # fly has reached the center point. Otherwise you will have 
        # to do some zero-division, which can be bad. 
        imin, dtmin, interceptpt_min = 0, np.inf, None
        for idx in range(numflies):

            if flies_collected_p[idx] == False:
                
                current_fly_posn = fly_trajs[idx][-1]
                dt_test, _ , interceptpt_test = intercept_info(current_horse_posn, current_fly_posn, 
                                                                phi, center, flies_reached_center_p[idx])
                #print Fore.YELLOW, dt_test, "------", dtmin, Style.RESET_ALL
                if dt_test < dtmin:
                    dtmin, imin, interceptpt_min  = dt_test, idx, interceptpt_test

        #print "=============================="
        #print dtmin, imin, interceptpt_min

        # Move the horse and the uncollected  flies by the time dt. Here 
        # Flies continue towards the center, and if they reach there they 
        # wait. Update the array marking whether a drone has reached the 
        # center or not.

        # First move the horse
        horse_traj.append({'coords'            : interceptpt_min, 
                           'fly_idxs_picked_up': [imin]         , # exactly one fly is picked up in each stage. 
                           'waiting_time'      : 0.0} )           # in this particular scheme the horse never waits

        # Now move the uncollected flies
        for idx in range(numflies):
            if flies_collected_p[idx] == False:
                current_fly_posn = fly_trajs[idx][-1]
                time_to_center   = np.linalg.norm(center-current_fly_posn)/phi
                
                if dtmin < time_to_center: # center not reached in this step, just do the normal vector advance
                    v      = center-current_fly_posn
                    unit_v = 1.0/np.linalg.norm(v) * v
                    new_fly_posn = current_fly_posn + dtmin * phi * unit_v

                else:  # the fly reaches the center here. stop the motion of the fly at the center
                    new_fly_posn                = center
                    flies_reached_center_p[idx] = True
                
                fly_trajs[idx].append(new_fly_posn)
                
        # Flip the boolean flag on the fly which was just collected
        flies_collected_p[imin] = True

    utils_algo.print_list(horse_traj)
    
    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(phi                = phi, 
                     horse_trajectories = [horse_traj],
                     fly_trajectories   = fly_trajs,
                     animation_file_name_prefix = None,
                     algo_name = 'gkin')
    
    return horse_traj, fly_trajs

        

#------------------------------------------------------------------------------------------------------------------------------   
def algo_greedy_nn_concentric_routing(sites, inithorseposn, phi, \
                                      shortcut_squiggles_p        = True,\
                                      write_algo_states_to_disk_p = False,\
                                      write_io_p                  = True,\
                                      animate_tour_p              = True,\
                                      plot_tour_p                 = False) :
    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    from sklearn.neighbors import NearestNeighbors

    if write_io_p:
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
    horse_traj          = [ {'coords'            : np.asarray(inithorseposn), 
                             'fly_idxs_picked_up': []                       , 
                             'waiting_time'      : 0.0} ] 
    fly_trajs           = [ [np.array(sites[i])] for i in range(numsites)] 
    flies_collected_p   = [False for i in range(numsites)]

    # There are exactly $n$ iterations in this loop where exactly 
    # one fly is picked up by the horse in each iteration. 
    while not(all(flies_collected_p)):
        current_horse_posn   = horse_traj[-1]['coords']
        unserviced_flies_idx = [idx for idx in range(len(flies_collected_p)) 
                                 if flies_collected_p[idx] == False ]
        fly_trajs_unserviced = [fly_trajs[idx] for idx in unserviced_flies_idx]

        imin = 0
        dmin = np.inf
        for idx in unserviced_flies_idx:
            current_fly_posn = fly_trajs[idx][-1]
            dmin_test = np.linalg.norm(current_fly_posn-current_horse_posn)
            if dmin_test < dmin:
                imin = idx
                dmin =  dmin_test

        # the meeting point of the horse and the fly closest to the horse
        horse_nearest_fly_distance     = np.linalg.norm(current_horse_posn-fly_trajs[imin][-1])
        horse_fly_uvec                 = 1.0/horse_nearest_fly_distance * (fly_trajs[imin][-1]-current_horse_posn) 
        dist_of_horse_to_meeting_point = 1.0/(phi+1.0) * horse_nearest_fly_distance
        center_heading                 = current_horse_posn + dist_of_horse_to_meeting_point * horse_fly_uvec
       
        # Move the horse and flies
        # In the greedy algorithm exactly one fly is collected
        # in each iteration
        [subidx], \
        horse_wait_time   , \
        new_fly_posns =     \
              head_towards_center(center_heading     = center_heading, 
                                  current_horse_posn = current_horse_posn, 
                                  current_fly_posns  = [traj[-1] for traj in fly_trajs_unserviced], 
                                  phis               = [phi for i in range(len(fly_trajs_unserviced))]) 

        # update the fly trajectories of the unserviced flies
        for idx, posn in zip(unserviced_flies_idx, new_fly_posns):
            fly_trajs[idx].append(posn)  

        # Append a point to the horse trajectory
        horse_traj.append({'coords'             : center_heading,
                          'fly_idxs_picked_up'  : [unserviced_flies_idx[subidx]],
                          'waiting_time'        : horse_wait_time} ) 

        # Flip the boolean flag on the fly which was just collected
        flies_collected_p[unserviced_flies_idx[subidx]] = True


    # shortcut the squiggles of each fly trajectory so that each 
    # fly heads directly towards its pick up point and waits for
    # the horse to come and pick it up. However to do this 
    # shortcutting, you should not give as the answer just the start point 
    # and end point of each trajectories, you need to record
    # where on the ray the fly is when the horse is picking up 
    # some fly. All points recorded on one shortcutted trajectory 
    # are collinear. These intermediate points don't seem to be useful 
    # from the point of view of analysis: however, they are extremely useful
    # during the animation. Note that the shortcutting used here does 
    # not change the makespan at all. It just makes it more convenient 
    # for the drones to just go to a designated target and wait. 
    if shortcut_squiggles_p:
       new_fly_trajs = [ ]

       for traj in fly_trajs:
           startpt    = traj[0]
           endpt      = traj[-1]

           v       = endpt - startpt
           unit_v  = 1.0/np.linalg.norm(v) * v
           newtraj = [startpt]
           
           for hidx in range(len(horse_traj)-1):
               currentpt = newtraj[-1]

               # Remember horse speed is assumed to be 1.0. 
               # Hence the curious division by 1.0 
               dt = np.linalg.norm(horse_traj[hidx+1]['coords']-\
                                   horse_traj[hidx]['coords'])/1.0

               if dt*phi < np.linalg.norm(currentpt-endpt):
                   newtraj.append( currentpt + dt*phi*unit_v   )
               else: 
                   newtraj.append( endpt )  
                   break # shortcutting is complete! break out of
                         # the loop and process next trajectory 
                   
           new_fly_trajs.append(newtraj)

       fly_trajs = new_fly_trajs

    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(phi                = phi, 
                     horse_trajectories = [horse_traj],
                     fly_trajectories   = fly_trajs,
                     animation_file_name_prefix = dir_name + '/' + io_file_name,
                     algo_name = 'gncr')
    
    return horse_traj, fly_trajs


#-----------------------------------------------------------------------------------------------
def algo_greedy_nn_concentric_routing_multiple_horses(sites, inithorseposns, phi,         \
                                                      shortcut_squiggles_p        = False,\
                                                      write_algo_states_to_disk_p = False,\
                                                      write_io_p                  = False,\
                                                      animate_tour_p              = True,\
                                                      plot_tour_p                 = False) :
    """ When more than one horse is given to us. Here we generalize the idea
    of greedy nn concentric routing. Note that each horse has speed 1.0
    """

    numflies      = len(sites)
    numhorses     = len(inithorseposns)

    horse_trajs   = [[{'coords'            : posn, 
                       'fly_idxs_picked_up': []  , 
                       'waiting_time'      : 0.0} ] for posn in inithorseposns ] 

    fly_trajs         = [ [np.array(sites[i])] for i in range(numflies)] 
    flies_collected_p = [False for i in range(numflies)]

    while not(all(flies_collected_p)):

        uncollected_flies_idx = [idx for idx in range(len(flies_collected_p)) 
                                 if flies_collected_p[idx] == False ]

        # Each horse finds the nearest uncollected fly that it could 
        # have captured and heads towards the interception point
        # assuming that fly would have moved towards the horse.
        
        interception_fidxs = [] # this list records for each horse the index of the nearest fly
        interception_pts   = [] # this list records the corresponding interception points if the 
                                # horse and the nearest fly had headed for one another. 

        for htraj in horse_trajs:
            current_horse_posn = htraj[-1]['coords']
            
            # Find the index of the nearest fly and the corresponding
            # interception point *if* the fly and the horse
            # had headed towards one another. 

            imin , dmin = 0, np.inf
            for fidx in uncollected_flies_idx:

                current_fly_posn = fly_trajs[fidx][-1]
                dist_to_horse    = np.linalg.norm(current_horse_posn-\
                                                  current_fly_posn)

                if  dist_to_horse < dmin:
                    imin = fidx
                    dmin = dist_to_horse
             
            # calculate the hypothetical interception point
            horse_nearest_fly_distance     = np.linalg.norm(current_horse_posn - fly_trajs[imin][-1])
            horse_fly_uvec                 = 1.0/horse_nearest_fly_distance * (fly_trajs[imin][-1]-current_horse_posn) 
            dist_of_horse_to_meeting_point = 1.0/(phi+1.0) * horse_nearest_fly_distance
            interception_pt                = current_horse_posn + dist_of_horse_to_meeting_point * horse_fly_uvec

            interception_fidxs.append(imin)
            interception_pts.append(interception_pt)

        assert(len(interception_fidxs) == numhorses)
        assert(len(interception_pts)   == numhorses)
        
        # Each fly on the other hand finds the nearest interception 
        # point computed above and heads towards that interception point

        # The first fly to reach an interception point is marked as collected. 
        # note that it will reach simultaneously with some horse that collects it. 
        # This will be the nearest horse-fly pair, exactly like nearest 
        # bichromatic pair. We calculate the time dt it takes for this fly to 
        # reach the horse positions for all horses and flies are updated by this 
        # time dt. Also there is no waiting by any of the horses or flies. 
        pass




        
#--------------------------------------------------------------------------------------------------
def head_towards_center(center_heading,        \
                        current_horse_posn,    \
                        current_fly_posns,\
                        phis):
    """ Given a horse (speed 1) and a bunch of flies (speed phis) and a center-heading, 
    all animals move towards the center-heading. If the horse gets to the center-heading
    before any flies, it waits till it meets the first fly. 
    (In this case, the waiting time for the horse is recorded along with the index of the fly)

    If one or more flies get to the center-heading before the horse, they wait there for the horse
    to come and pick them up. (The indices of the flies picked up are recorded. The waiting time 
    for the horse is zero.)

    The answer returned is 
    1. The indices of the flies picked up 
    2. The waiting time for the horse at the center
    3. The new positions of the flies
    
    The new position of the horse is just the center heading.(duh!)
    """
    numflies = len(current_fly_posns)
    assert(len(phis) == numflies)

    horse_time_to_center          =  np.linalg.norm( current_horse_posn - center_heading )/1.0
    fly_times_to_center_with_idx  = zip([np.linalg.norm( current_fly_posns[idx] -
                                                center_heading )/phis[idx] \
                                             for idx in range(numflies)], 
                                        range(numflies))

    # Find the list of flies who reached the center before the horse
    fly_idxs_reached_center_before_horse = [ tup[1] for tup in fly_times_to_center_with_idx
                                             if tup[0] <= horse_time_to_center  ]

    if fly_idxs_reached_center_before_horse:
        # at least one fly reached the center before the horse
        fly_idxs_picked_up = fly_idxs_reached_center_before_horse
        dt                 = horse_time_to_center
        horse_wait_time    = 0.0             

        new_fly_posns      = []
        for i in range(numflies):

            if i in fly_idxs_reached_center_before_horse:
                new_fly_posns.append(center_heading)
            else:
                v      = center_heading - current_fly_posns[i]
                unit_v =  1/np.linalg.norm(v) * v
                new_fly_posns.append(current_fly_posns[i] + dt*phis[i]*unit_v)

    else:                                    
        # the horse reached before all the flies
        fly_times_to_center       = [tup[0] for tup in fly_times_to_center_with_idx]
        idx_fastest_fly_to_center =  min(xrange(len(fly_times_to_center)), key=fly_times_to_center.__getitem__) # https://stackoverflow.com/a/11825864
        horse_wait_time           =  min(fly_times_to_center) - horse_time_to_center
        assert(horse_wait_time>=0)

        fly_idxs_picked_up = [idx_fastest_fly_to_center] 
        dt                 = min(fly_times_to_center)

        new_fly_posns      = []
        for i in range(numflies):
                v      = center_heading - current_fly_posns[i]
                unit_v =  1/np.linalg.norm(v) * v
                new_fly_posns.append(current_fly_posns[i] + dt*phis[i]*unit_v)

    #utils_algo.print_list(fly_idxs_picked_up)
    #print "\n", horse_wait_time
    #utils_algo.print_list(new_fly_posns)
    return fly_idxs_picked_up, horse_wait_time, new_fly_posns


#----------------------------------------------------------------------------------------------

def intercept_info(current_horse_posn, current_fly_posn, phi, center, center_reached_p):
    
    if center_reached_p: 
        dt                   = np.linalg.norm(current_fly_posn-current_horse_posn)/1.0 # all horses have speed 1.0
        fly_reaches_center_p = True # vacuously true, since the fly is already at the center
        interceptpt          = center

    else: # fly has not reached center
 
        # vector joining current_fly_posn to center
        u      = center - current_fly_posn 
        unit_u = 1.0/np.linalg.norm(u) * u
        
        # vector joining current_fly_posn to current_horse_posn
        v      = current_horse_posn - current_fly_posn
        unit_v = 1.0/np.linalg.norm(v) * v

        cos_alpha          = np.dot(unit_u, unit_v) # cosine of the angle between u and v
        l                  = np.linalg.norm(v)      # distance between current_horse_posn and current_fly_posn
        fly_time_to_center = np.linalg.norm(center-current_fly_posn)/phi

        discriminant = 4.0 * l**2 * (cos_alpha**2 - (1.0 - 1.0/phi**2))  
        assert(discriminant>0.0)

        if discriminant < 0.0:  #### HERE BE DRAGONS discriminant <0.0 seems a troublesome case. 
            fly_reaches_center_p = True
            dt                   = np.linalg.norm(center - current_horse_posn)/1.0
            interceptpt          = center

        else:
            roots  = np.roots([(1.0-1.0/phi**2), -2.0*l*cos_alpha, l**2])
            assert(all(map(np.isreal, roots)))
            #print Fore.GREEN, "----->", roots, Style.RESET_ALL
            assert(max(roots)>0.0)
            
            if max(roots)/phi > fly_time_to_center: # fly reaches the center before the intercept happens. So it just waits there
                fly_reaches_center_p = True
                dt                   = np.linalg.norm(center - current_horse_posn)/1.0
                interceptpt          = center

            else :                                 # intercept happens, before the fly reaches the center. 
                fly_reaches_center_p = False
                dt                   = max(roots)/phi
                interceptpt          = current_fly_posn + dt * phi * unit_u
               
    return dt, fly_reaches_center_p, interceptpt
            

#---------------------------------------------------------------------------------------------


def plot_tour_gncr (sites, inithorseposn, phi, \
               horse_trajectory,               \
               fly_trajectories,               \
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
                  animation_file_name_prefix, algo_name,  render_trajectory_trails_p = False):
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

    mspan, _ = makespan(horse_trajectories)
    ax.set_title("Algo: " + algo_name + "  Makespan: " + '%.4f' % mspan , fontsize=25)

    number_of_flies  = len(fly_trajectories)
    number_of_horses = len(horse_trajectories)
    colors           = utils_graphics.get_colors(number_of_horses, lightness=0.5)
        
    ax.set_xlabel( "Number of drones: " + str(number_of_flies) + "\n" + r"$\varphi=$ " + str(phi), fontsize=25)
    ims                = []
    
    # Constant for discretizing each segment inside the trajectories of the horses
    # and flies. 
    NUM_SUB_LEGS              = 2 # Number of subsegments within each segment of every trajectory
    
    # Arrays keeping track of the states of the horses
    horses_reached_endpt_p    = [False for i in range(number_of_horses)]
    horses_traj_num_legs      = [len(traj)-1 for traj in horse_trajectories] # the -1 is because the initial position of the horse is always included. 
    horses_current_leg_idx    = [0 for i in range(number_of_horses)]
    horses_current_subleg_idx = [0 for i in range(number_of_horses)] 
    horses_current_posn       = [traj[0]['coords'] for traj in horse_trajectories]

    # List of arrays keeping track of the flies collected by the horses at any given point in time, 
    fly_idxs_collected_so_far = [[] for i in range(number_of_horses)] 

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
                htraj                       = [elt['coords'] for elt in horse_trajectories[hidx]]
                all_flys_collected_by_horse = [i             for elt in horse_trajectories[hidx] for i in elt['fly_idxs_picked_up']]

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

                ####################......for marking in yellow during rendering
                fly_idxs_collected_so_far[hidx].extend(horse_trajectories[hidx][legidx]['fly_idxs_picked_up'])
                fly_idxs_collected_so_far[hidx] = list(set(fly_idxs_collected_so_far[hidx])) ### critical line, to remove duplicate elements # https://stackoverflow.com/a/7961390

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
            traj               = [elt['coords'] for elt in horse_trajectories[hidx]]
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

            if render_trajectory_trails_p:
                flyline, = ax.plot(xfs,yfs,'-',linewidth=2.5, markersize=6, alpha=0.2, color='b')
                objs.append(flyline)


            # If the current fly is in the list of flies already collected by some horse, 
            # mark this fly with yellow. If it hasn't been serviced yet, mark it with blue
            service_status_col = 'b'
            for hidx in range(number_of_horses):
                #print fly_idxs_collected_so_far[hidx]
                if fidx in fly_idxs_collected_so_far[hidx]:
                    service_status_col = 'y'
                    break

            flyloc   = Circle((current_fly_posn[0], current_fly_posn[1]), 0.009, 
                              facecolor = service_status_col, edgecolor='k', alpha=1.00)
            flypatch = ax.add_patch(flyloc)
            objs.append(flypatch)
        
        print "........................"
        print "Appending to ims ", image_frame_counter
        ims.append(objs) 
        image_frame_counter += 1

    from colorama import Back 
    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    plt.show()




def makespan(horse_trajectories):
    """ 
    The function returns the makespan along with the index 
    of the trajectory that gives this makespan. 

    The time to completion of a single horsetrajectory is the 
    length of the polygonal chain plus the waiting times at each 
    of the vertices of the chain. Remember all horses are assumed 
    to have the same speed i.e. 1.0 so it makes sense to add 
    'length' to 'time'. 

    The makespan of a set of horse trajectories is the maximum 
    of the time to completion of each of the horse trajectories. 
    This is the time taken for *all* the flies to be collected. 
    """
    
    times_to_completion = []

    for traj in horse_trajectories:
        tour_length        = utils_algo.length_polygonal_chain( [elt['coords'] for elt in traj] )
        total_waiting_time = sum( [elt['waiting_time'] for elt in traj] )
        times_to_completion.append(tour_length + total_waiting_time)

    makespan  = max(times_to_completion)
    index_max = max(xrange(len(times_to_completion)), key=times_to_completion.__getitem__)
    return makespan, index_max
