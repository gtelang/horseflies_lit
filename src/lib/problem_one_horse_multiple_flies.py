    
# Relevant imports
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
#plt.style.use('seaborn-poster')
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time
import utils_algo
import utils_graphics

import problem_classic_horsefly as chf

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
                    # Set speed and number of flies
                    
                    phi_str = raw_input(Fore.YELLOW + "What should I set the speed of each of the flies to be (should be >1)? : " + Style.RESET_ALL)
                    nof_str = raw_input(Fore.YELLOW + "How many flies do you want me to assign to the horse? : " + Style.RESET_ALL)

                    phi = float(phi_str)
                    nof = int(nof_str)
                    
                    # Select algorithm to execute

                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (ec)   Earliest Capture \n"                                        +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'ec':
                          tour = run.getTour( algo_greedy_earliest_capture, phi, \
                                              number_of_flies = nof)
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
    run = MultipleFliesInput()
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
class MultipleFliesInput:
      def __init__(self, sites=[], inithorseposn=()):
           self.sites           = sites
           self.inithorseposn   = inithorseposn

      # Methods for \verb|MultipleFliesInput|
      def clearAllStates (self):
         self.sites = []
         self.inithorseposn = ()
      def getTour(self, algo, speedratio, number_of_flies):
            return algo(self.sites, self.inithorseposn, speedratio, number_of_flies)
      


# Algorithms for multiple flies

# Helper functions for \verb|algo_greedy_earliest_capture|
   
def meeting_time_horse_fly_opp_dir(horseposn, flyposn, flyspeed):
    horseposn = np.asarray(horseposn)
    flyposn   = np.asarray(flyposn)
    return 1/(flyspeed+1) * np.linalg.norm(horseposn-flyposn)
    
# Definition of the \verb|FlyState| class
class FlyState:
    def __init__(self, idx, initflyposn, site, flyspeed):

         self.idx                                = idx
         self._flytraj                           = [ {'coordinates': np.asarray(initflyposn), 'type':'gen_pt'} ]
         self._current_assigned_site             = np.asarray(site)
         self._speed                             = flyspeed
         self._current_assigned_site_serviced_p  = False
         self._fly_retired_p                     = False
    
    def retire_fly(self):
         self._fly_retired_p = True
 
    def deploy_to_site(self,site):
         self._current_assigned_site            = np.asarray(site)
         self._current_assigned_site_serviced_p = False 

    def is_retired(self):
         return self._fly_retired_p

    def is_current_assigned_site_serviced(self):
         return self._current_assigned_site_serviced_p

    def get_current_fly_position(self):
         return self._flytraj[-1]['coordinates']
   
    def get_trajectory(self):
         return self._flytraj

    # Definition of method \verb|update_fly_trajectory|
    
    def update_fly_trajectory(self, dt, rendezvous_pt):

         if self.is_retired():
            return 

         dx = self._speed * dt

         if self._current_assigned_site_serviced_p :
            # Move towards the provided rendezvous point
               
            heading  = rendezvous_pt - self.get_current_fly_position()
            uheading = heading / np.linalg.norm(heading) 
            newpt    = self.get_current_fly_position() + dx * uheading
            self._flytraj.append(  {'coordinates': newpt, 'type': 'gen_pt'}  )
            

         elif dx < np.linalg.norm(self._current_assigned_site - self.get_current_fly_position()) :
            # Continue moving towards the site
            
            heading  = self._current_assigned_site - self.get_current_fly_position()
            uheading = heading / np.linalg.norm(heading) 
            newpt    = self.get_current_fly_position() + dx * uheading
            self._flytraj.append(  {'coordinates': newpt, 'type': 'gen_pt'}  )
            
         else: 
            # Move towards the site mark site as serviced and then head towards rendezvous point
            
            dx_reduced = dx - np.linalg.norm(self._current_assigned_site -\
                                             self.get_current_fly_position())
            heading  = rendezvous_pt - self._current_assigned_site
            uheading = heading/np.linalg.norm(heading)

            newpt = self._current_assigned_site + uheading * dx_reduced
            self._current_assigned_site_serviced_p = True
            self._flytraj.extend([{'coordinates':self._current_assigned_site, 'type':'site'}, 
                                  {'coordinates':newpt,                       'type':'gen_pt'}])
            
     
    # Definition of method \verb|rendezvous_time_and_point_if_selected_by_horse|
    
    def rendezvous_time_and_point_if_selected_by_horse(self, horseposn):
       assert(self._fly_retired_p != True)
      
       if self._current_assigned_site_serviced_p:
           rt = meeting_time_horse_fly_opp_dir(horseposn, self.get_current_fly_position(), self._speed)
           horseheading = self.get_current_fly_position() - horseposn
       else:
          distance_to_site    = np.linalg.norm(self.get_current_fly_position() -\
                                               self._current_assigned_site)
          time_of_fly_to_site = 1/self._speed * distance_to_site

          horse_site_vec   = self._current_assigned_site - horseposn 
          displacement_vec = time_of_fly_to_site * horse_site_vec/np.linalg.norm(horse_site_vec)
          horseposn_tmp   = horseposn + displacement_vec

          time_of_fly_from_site = \
                   meeting_time_horse_fly_opp_dir(horseposn_tmp, self._current_assigned_site, self._speed)

          rt = time_of_fly_to_site + time_of_fly_from_site
          horseheading = self._current_assigned_site - horseposn

       uhorseheading = horseheading/np.linalg.norm(horseheading)
       return rt, horseposn + uhorseheading * rt

    

def algo_greedy_earliest_capture(sites, inithorseposn, phi, number_of_flies,\
                                 write_algo_states_to_disk_p = True,\
                                 write_io_p                  = True,\
                                 animate_tour_p              = True) :

    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    algo_name     = 'algo-greedy-earliest-capture'
    time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    dir_name      = algo_name + '---' + time_stamp
    io_file_name  = 'input_and_output.yml'

    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    algo_state_counter = 1 
    

    if number_of_flies > len(sites):
          number_of_flies = len(sites)

    current_horse_posn = np.asarray(inithorseposn)
    horse_traj         = [(current_horse_posn, None)]

    # Find the $k$-nearest sites to \verb|inithorseposn| for $k=$\verb|number_of_flies| and claim them
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=number_of_flies)
    neigh.fit(sites)

    _, knn_idxss = neigh.kneighbors([inithorseposn])
    knn_idxs     = knn_idxss.tolist()[0]
    knns         = [sites[i] for i in knn_idxs]
    unclaimed_sites_idxs = list(set(range(len(sites))) - set(knn_idxs)) # https://stackoverflow.com/a/3462160
    
    # Initialize one \verb|FlyState| object per fly for all flies
    flystates = []
    for i in range(number_of_flies):
        flystates.append(FlyState(i,inithorseposn, knns[i], phi))
    

    all_flies_retired_p = False

    while (not all_flies_retired_p):
       # Find the index of the fly \bm{F} which can meet the horse at the earliest, the rendezvous point $R$, and time till rendezvous
       imin  = 0
       rtmin = np.inf
       rptmin= None
       for i in range(number_of_flies):
            if flystates[i].is_retired():
                continue
            else:
                rt, rpt = flystates[i].rendezvous_time_and_point_if_selected_by_horse(current_horse_posn)
                if rt < rtmin:
                    imin   = i
                    rtmin  = rt
                    rptmin = rpt
       
       # Update fly trajectory in each \verb|FlyState| object till \bm{F} meets the horse at $R$
       for flystate in flystates:
           flystate.update_fly_trajectory(rtmin, rptmin)
        
       # Update \verb|current_horse_posn| and horse trajectory
       current_horse_posn = rptmin
       horse_traj.append((np.asarray(rptmin),imin))
       
       # Deploy \bm{F} to an unclaimed site if one exists and claim that site, otherwise retire \bm{F}
         
       if  unclaimed_sites_idxs:
           unclaimed_sites = [sites[i] for i in unclaimed_sites_idxs]

           neigh = NearestNeighbors(n_neighbors=1)
           neigh.fit(unclaimed_sites)

           _, nn_idxss = neigh.kneighbors([current_horse_posn])
           nn_idx      = nn_idxss.tolist()[0][0]

           flystates[imin].deploy_to_site(unclaimed_sites[nn_idx])
           unclaimed_sites_idxs = list(set(unclaimed_sites_idxs) - \
                                       set([unclaimed_sites_idxs[nn_idx]]))

       else: 
           flystates[imin].retire_fly()
        
       # Calculate value of \verb|all_flies_retired_p|
       acc = True 
       for i in range(number_of_flies):
            acc = acc and flystates[i].is_retired()
       all_flies_retired_p = acc
       
       # Write algorithms current state to file, if \verb|write_algo_states_to_disk_p == True|
       
       print "Algorithm State Number: ", algo_state_counter
       if write_algo_states_to_disk_p:
            algo_state_file_name = 'algo_state_' + str(algo_state_counter).zfill(5) + '.yml'

            data = {'horse_trajectory' : horse_traj, \
                    'fly_trajectories' : [flystates[i].get_trajectory() for i in range(number_of_flies)] }
            utils_algo.write_to_yaml_file(data, dir_name=dir_name, file_name=algo_state_file_name)
       algo_state_counter += 1
        
    
    # Write input and output to file if \verb|write_io_p == True|
    if write_io_p:
         data = { 'sites' : sites, \
                  'inithorseposn' : inithorseposn,\
                  'phi':phi,\
                  'horse_trajectory' : horse_traj, \
                  'fly_trajectories' : [flystates[i].get_trajectory() 
                                       for i in range(number_of_flies)] }
         utils_algo.write_to_yaml_file(data, dir_name = dir_name, file_name = io_file_name)
    
    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(sites            = sites, 
                     inithorseposn    = inithorseposn, 
                     phi              = phi, 
                     horse_trajectory = horse_traj, 
                     fly_trajectories = [flystates[i].get_trajectory() for i in range(number_of_flies)],
                     animation_file_name_prefix = dir_name + '/' + io_file_name)
    
    # Return multiple flies tour
    return {'sites' : sites, \
              'inithorseposn' : inithorseposn,\
              'phi':phi,\
              'horse_trajectory': horse_traj, \
              'fly_trajectories': [flystates[i].get_trajectory() for i in range(number_of_flies)]}
    


# Plotting routines

def plot_tour(ax, tour):

    sites            = tour['sites']
    inithorseposn    = tour['inithorseposn']
    phi              = tour['phi']
    horse_trajectory = tour['horse_trajectory']
    fly_trajectories = tour['fly_trajectories']

    xhs = [ horse_trajectory[i][0][0] for i in range(len(horse_trajectory))]    
    yhs = [ horse_trajectory[i][0][1] for i in range(len(horse_trajectory))]    

    number_of_flies = len(fly_trajectories)
    colors          = utils_graphics.get_colors(number_of_flies, lightness=0.4)

    ax.cla()
    utils_graphics.applyAxCorrection(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot fly trajectories
    xfss = [[point['coordinates'][0] for point in fly_trajectories[i]] for i in range(len(fly_trajectories))]
    yfss = [[point['coordinates'][1] for point in fly_trajectories[i]] for i in range(len(fly_trajectories))]
 
    for xfs, yfs,i in zip(xfss,yfss,range(number_of_flies)):
        ax.plot(xfs,yfs,color=colors[i], alpha=0.7)

    # Plot sites along each flys tour
    xfsitess = [ [point['coordinates'][0] for point in fly_trajectories[i] if point['type'] == 'site'] 
                for i in range(len(fly_trajectories))]
    yfsitess = [ [point['coordinates'][1] for point in fly_trajectories[i] if point['type'] == 'site'] 
                for i in range(len(fly_trajectories))]
    
    for xfsites, yfsites, i in zip(xfsitess, yfsitess, range(number_of_flies)):
        for xsite, ysite, j in zip(xfsites, yfsites, range(len(xfsites))):
              ax.add_patch(mpl.patches.Circle((xsite,ysite), radius = 1.0/140, \
                                              facecolor=colors[i], edgecolor='black'))
              ax.text(xsite, ysite, str(j+1), horizontalalignment='center', 
                                              verticalalignment='center'  , 
                                              bbox=dict(facecolor=colors[i], alpha=1.0)) 
    # Plot horse tour
    ax.plot(xhs,yhs,'o-',markersize=5.0, linewidth=2.5, color='#D13131') 
    
    # Plot initial horseposn 
    ax.add_patch( mpl.patches.Circle( inithorseposn,radius = 1.0/100,
                                    facecolor= '#D13131', edgecolor='black'))


# Animation routines
   
def animate_tour (sites, inithorseposn, phi, horse_trajectory, fly_trajectories, animation_file_name_prefix):
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
    colors          = utils_graphics.get_colors(number_of_flies, lightness=0.5)

    ax.set_title("Number of sites: " + str(len(sites)), fontsize=25)
    ax.set_xlabel(r"$\varphi=$ " + str(phi) + "\nNumber of flies: " + str(number_of_flies), fontsize=20)
    
    # Parse trajectory information and convert trajectory representation to leg list form
     
    # Leg list form for all horse trajectories
    horse_traj_ll = []
    for i in range(len(horse_trajectory)-1):
        horse_traj_ll.append((horse_trajectory[i][0], horse_trajectory[i+1][0], 
                              horse_trajectory[i+1][1]))

    # Leg list form for all fly trajectories
    fly_trajs_ll = []
    for fly_traj in fly_trajectories:
        fly_traj_ll = []
        for i in range(len(fly_traj)-1):
            if fly_traj[i]['type'] == 'gen_pt':
     
                 if fly_traj[i+1]['type'] == 'gen_pt':
                      fly_traj_ll.append((fly_traj[i], 
                                          fly_traj[i+1]))
                      
        
                 elif fly_traj[i+1]['type'] == 'site':
                      fly_traj_ll.append((fly_traj[i], \
                                          fly_traj[i+1], \
                                          fly_traj[i+2]))
        fly_trajs_ll.append(fly_traj_ll)

    num_horse_legs = len(horse_traj_ll)

    # Append empty legs to fly trajectories so that leg counts 
    # for all fly trajectories are the same as that of the horse
    # trajectory
    for fly_traj in fly_trajs_ll:
        m = len(fly_traj)
        empty_legs = [None for i in range(num_horse_legs-len(fly_traj))]
        fly_traj.extend(empty_legs)

    
    # Construct and store every frame of the animation in the \verb|ims| array
       
    # Define discretization function for a leg of the horse or fly tour

    def discretize_leg(pts):
        subleg_pts = []

        if pts == None:
             return None
        else:
             numpts = len(pts)

             if numpts == 2:   # horse leg or fly-leg of type gg
                 k  = 19 
                 legtype = 'gg'
             elif numpts == 3: # fly leg of type gsg 
                 k  = 10 
                 legtype = 'gsg'

             pts = map(np.asarray, pts)
             for p,q in zip(pts, pts[1:]):
                 tmp = []
                 for t in np.linspace(0,1,k): 
                       tmp.append((1-t)*p + t*q) 
                 subleg_pts.extend(tmp[:-1])

             subleg_pts.append(pts[-1])
             return {'points': subleg_pts, 
                     'legtype'  : legtype}
    
    ims                = []
    horse_points_so_far = []
    fly_points_so_far   = [[] for i in range(number_of_flies)] 
    fly_sites_so_far    = [[] for i in range(number_of_flies)] # each list is definitely a sublist of corresponding list in fly_points_so_far
    for idx in range(len(horse_traj_ll)):
        # Get current horse-leg and update the list of points covered so far by the horse
        horse_leg = (horse_traj_ll[idx][0], horse_traj_ll[idx][1])
        horse_points_so_far.append(horse_leg[0]) # attach the beginning point of the horse leg
        horse_leg_pts = horse_leg
        #utils_algo.print_list(horse_points_so_far)
        #print "....................................................."

        fly_legs  = [fly_trajs_ll[i][idx] for i in range(len(fly_trajs_ll)) ]
        fly_legs_pts  = []
        for fly_leg, i in zip(fly_legs, range(len(fly_legs))):
           if fly_leg != None:
                coods = []
                for pt in fly_leg:
                     coods.append(pt['coordinates'])
                fly_legs_pts.append(coods)
                fly_points_so_far[i].append(coods[0]) # attaching the beginning point of the leg. Extension only 
                                                      # happens for legs wqhich are not of type None, meshing well 
                                                      # with the fact that fly has stopped moving. 
           else:
                fly_legs_pts.append(None)

        # discretize current leg, for horse and fly, and for each point in the discretization
        # render the frame. If a fly crosses a site, update the fly_points_so_far list
        horse_leg_disc = discretize_leg(horse_leg_pts)   # list of points 
        fly_legs_disc  = map(discretize_leg, fly_legs_pts) # list of list of points 

        # Each iteration of the following loop tacks on a new frame to ims
        # this outer level for loop is just proceeding through each position
        # in the discretized horse legs. This is the motion which coordinates
        # the fly's motions
        for k in range(len(horse_leg_disc['points'])):
            current_horse_posn = horse_leg_disc['points'][k]
            current_fly_posns  = []  # updated in the for loop below.
            for j in range(len(fly_legs_disc)):
                  if fly_legs_disc[j] != None:
                        current_fly_posns.append(fly_legs_disc[j]['points'][k])

                        if fly_legs_disc[j]['legtype'] == 'gsg' and k==9: # yay, we just hit a site!
                              fly_points_so_far[j].append(fly_legs_disc[j]['points'][k])
                              fly_sites_so_far[j].append(fly_legs_disc[j]['points'][k])
                  else: 
                        current_fly_posns.append(None)
            objs = []
            # Plot sites as black circles
            for site in sites:
                 circle = Circle((site[0], site[1]), 0.01, \
                                  facecolor = 'k'   , \
                                  edgecolor = 'black'     , \
                                  linewidth=1.0)
                 sitepatch = ax.add_patch(circle)
                 objs.append(sitepatch)

            # Plot trajectory of horse
            xhs = [pt[0] for pt in horse_points_so_far] + [current_horse_posn[0]]
            yhs = [pt[1] for pt in horse_points_so_far] + [current_horse_posn[1]]
            horseline, = ax.plot(xhs,yhs,'r-',linewidth=5.0, markersize=6, alpha=1.00)
            objs.append(horseline)
            # Plot trajectory of flies (no markers)
            assert(len(fly_points_so_far) == number_of_flies)
            for ptraj, i in zip(fly_points_so_far, range(number_of_flies)):
                 print current_fly_posns[i]
                 if current_fly_posns[i] is None:
                       xfs = [pt[0] for pt in ptraj]
                       yfs = [pt[1] for pt in ptraj] 
                 else:
                       xfs = [pt[0] for pt in ptraj] + [current_fly_posns[i][0]]
                       yfs = [pt[1] for pt in ptraj] + [current_fly_posns[i][1]]

                 flyline, = ax.plot(xfs, yfs, '-', linewidth=2.5, alpha=0.30, color=colors[i])
                 objs.append(flyline)
            # Plot the end points of trajectories only
            # Plot currently covered sites as colored circles
            for sitelist, i in zip(fly_sites_so_far, range(number_of_flies)):
               for site in sitelist:
                     circle = Circle((site[0], site[1]), 0.015, \
                                      facecolor = colors[i]   , \
                                      edgecolor = 'black'     , \
                                      linewidth=1.0)
                     sitepatch = ax.add_patch(circle)
                     objs.append(sitepatch)


            #debug(Fore.CYAN + "Appending to ims "+ Style.RESET_ALL)
            ims.append(objs) # [::-1] means reverse the list

    # Write animation of tour to disk and display in live window
    from colorama import Back 

    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True, repeat_delay=1000)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=150)
    #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

    plt.show() # For displaying the animation in a live window. 
    
    # Write animation of tour to disk and display in live window
       
    from colorama import Back 

    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    ani.save(animation_file_name_prefix+'.avi', dpi=250)
    debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

    plt.show() # For displaying the animation in a live window. 
    

