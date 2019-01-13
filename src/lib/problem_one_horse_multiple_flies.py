    
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
                                 animate_schedule_p          = True):

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
    horse_traj         = [current_horse_posn]

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
       print "Algorithm State Number: ", algo_state_counter
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
       horse_traj.append(np.asarray(rptmin))
       
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
       
       if write_algo_states_to_disk_p:
            import yaml
            algo_state_file_name = 'algo_state_' + str(algo_state_counter).zfill(5) + '.yml'

            data = {'horse_trajectory' : horse_traj, \
                    'fly_trajectories' : [flystates[i].get_trajectory() 
                                          for i in range(number_of_flies)] }

            with open(dir_name + '/' + algo_state_file_name, 'w') as outfile:
                 yaml.dump( data, outfile, default_flow_style = False)
            algo_state_counter += 1
        
    
    # Write input and output to file if \verb|write_io_p == True|
    if write_io_p:
        print Fore.GREEN, "Horse Trajectory is ", Style.RESET_ALL
        utils_algo.print_list(horse_traj)
        for i in range(number_of_flies):
               print "Trajectory of Fly", i
               utils_algo.print_list(flystates[i].get_trajectory())
               print "----------------------------------------------"

        fig, ax =  plt.subplots()
        ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
        ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
        ax.set_aspect(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
      
        # Plot the fly trajectories
        # Place graphical elements in reverse order,
        # i.e. from answer, all the way upto question. 
        colors = utils_graphics.get_colors(number_of_flies)
        for i in range(number_of_flies):
           print i
           xfs = [pt['coordinates'][0] for pt in flystates[i].get_trajectory()]
           yfs = [pt['coordinates'][1] for pt in flystates[i].get_trajectory()]
           ax.plot(xfs,yfs, '-', linewidth=1.0, color=colors[i])
      
        # Plot the horse trajectory
        xhs = [ pt[0] for pt in horse_traj  ]
        yhs = [ pt[1] for pt in horse_traj  ]
        ax.plot(xhs,yhs, 'ro-',linewidth=3.0)

        # Plot sites
        xsites = [site[0] for site in sites]
        ysites = [site[1] for site in sites]
        ax.plot(xsites, ysites, 'bo')

        # Plot initial horseposition
        ax.plot([inithorseposn[0]], [inithorseposn[1]], 'ks', markersize=10.0)
        plt.show()
    
    sys.exit()






