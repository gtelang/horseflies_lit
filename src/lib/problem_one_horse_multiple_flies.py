    
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
# plt.style.use('seaborn-poster')
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
         self._flytraj                           = [ np.asarray(initflyposn) ]
         self._current_assigned_site             = np.asarray(site)
         self._speed                             = flyspeed
         self._current_assigned_site_serviced_p  = False
         self._fly_retired_p                     = False
    
    def retire_fly(self):
         self._fly_retired_p = True
 
    def redeploy_to_site(self,site):
         self._current_assigned_site            = np.asarray(site)
         self._current_assigned_site_serviced_p = False 

    def is_retired(self):
         return self._fly_retired_p

    def is_current_assigned_site_serviced(self):
         return self._current_assigned_site_serviced_p

    def get_current_fly_position(self):
         return self._flytraj[-1]

    # Definition of method \verb|update_fly_trajectory|
    
    def update_fly_trajectory(self, dt, rendezvous_pt):

         if self.is_retired():
            return 

         dx = self._speed * dt

         if self._current_assigned_site_serviced_p or \
                     (dx < np.linalg.norm( self._current_assigned_site -\
                                           self.get_current_fly_position())):   

              heading  = self.fly_traj[-1] - self.fly_traj[-2]
              uheading = heading / np.linalg.norm(heading) 
              newpt = self.fly_traj[-1] + dx * uheading
              self.fly_traj.append(newpt)

         else: # the fly needs to ``uturn'' at the site
              dx_reduced = dx - np.linalg.norm(self._current_assigned_site -\
                                               self.get_current_fly_position())
              assert(dx_reduced >= 0, "dx_reduced should be >= 0")
              heading  = rendezvous_pt - self._current_assigned_site
              uheading = heading/np.linalg.norm(heading)

              newpt = self._current_assigned_site + uheading * dx_reduced
              self.fly_traj.extend([self._current_assigned_site, newpt])
    
    # Definition of method \verb|rendezvous_time_and_point_if_selected_by_horse|
    
    def rendezvous_time_and_point_if_selected_by_horse(self, horseposn):
       assert(self._fly_retired_p != True)
      
       if self._current_assigned_site_serviced_p:
           rt = meeting_time_horse_fly_opp_dir(horseposn, \
                                               self.get_current_fly_position(),\
                                               self._speed)
           horseheading = self.get_current_fly_position()
       else:
          distance_to_site    = np.linalg.norm(self.get_current_fly_position() -\
                                               self._current_assigned_site)
          time_of_fly_to_site = 1/self._speed * distance_to_site

          horse_site_vec   = np.linalg.norm(self._current_assigned_site - horseposn) 
          displacement_vec = time_of_fly_to_site * horse_site_vec/np.linalg.norm(horse_site_vec)
          horse_posn_tmp   = horse_posn + displacement_vec

          time_of_fly_from_site = meeting_time_horse_fly_opp_dir(  \
                                      horseposn_tmp,               \
                                      self._current_assigned_site, \
                                      self._speed)

          rt = time_of_fly_to_site + time_of_fly_from_site
          horseheading = self._current_assigned_site

       uhorseheading = horseheading/np.linalg.norm(uhorseheading)
       return rt, horseposn + uhorseheading * rt

    
    # Definition of method \verb|print_current_state|
    
    def print_current_state(self):
        fly_speed_str = "Fly Speed is " + str(self._speed)                             
        fly_traj_str  = "Fly trajectory is " + ''.join(map(str, self._flytraj))             
        current_assigned_site_str = "Current assigned site is " +\
                                     str(self._current_assigned_site)             
        current_assigned_site_serviced_p_str = "Assigned site serviced: " +\
                                                str(self._current_assigned_site_serviced_p) 
        fly_retired_p_str = "Fly retired: " +  str(self._fly_retired_p)
        
        print '...................................................................'
        print Fore.BLUE    , fly_speed_str             , Style.RESET_ALL
        print Fore.MAGENTA , fly_traj_str              , Style.RESET_ALL
        print Fore.YELLOW  , current_assigned_site_str , Style.RESET_ALL
        print Fore.GREEN   , current_assigned_site_serviced_p_str, Style.RESET_ALL
        print Fore.RED     , fly_retired_p_str         , Style.RESET_ALL
    

def algo_greedy_earliest_capture(sites, inithorseposn, phi, number_of_flies,\
                                 write_algo_states_to_disk_p = True,\
                                 write_io_p                  = True,\
                                 animate_schedule_p          = True):

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
        flystates.append(FlyState(inithorseposn, knns[i], phi))
    

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
       horse_traj.append(np.asarray(rptmin))
       
       # Deploy \bm{F} to an unclaimed site if one exists and claim that site, otherwise retire \bm{F}
         
       if  unclaimed_sites_idxs:
           unclaimed_sites = [sites[i] for i in unclaimed_sites_idxs]

           neigh = NearestNeighbors(n_neighbors=1)
           neigh.fit(unclaimed_sites)

           _, nn_idxss = neigh.kneighbors([current_horse_posn])
           nn_idx      = nn_idxss.tolist()[0][0]

           assert( np.linalg.norm ( sites[unclaimed_sites_idxs[nn_idx]]  - \
                                    unclaimed_sites[nn_idx]  ) < 1e-8, \
                   "Assertion failure in deployment step" )

           flystates[imin].redeploy_to_site(unclaimed_sites[nn_idx])
           unclaimed_sites_idxs = list(set(unclaimed_sites_idxs) - \
                                       set([unclaimed_sites_idxs[nn_idx]]))

       else: # All sites have been claimed by some drone. There is no need for the fly anymore
           flystates[imin].retire_fly()
        
       # Calculate value of \verb|all_flies_retired_p|
       
       acc = True # accumulator variabvle
       for i in range(number_of_flies):
            acc and flystates[i].is_retired()
       all_flies_retired_p = tmp
       
       @<Write algorithms current state to file, if \verb|write_algo_states_to_disk_p == True|@> 
    
    @<Write input and output to file if \verb|write_io_p == True|@>
    @<Make an animation of the schedule if \verb|animate_schedule_p == True|@>





