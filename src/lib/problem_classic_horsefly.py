    
# Relevant imports for classic horsefly

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
                    phi_str = raw_input(Fore.YELLOW + "Enter speed of fly (should be >1): " + Style.RESET_ALL)
                    phi = float(phi_str)

                    input_str = raw_input(Fore.YELLOW                                            +\
                              "Enter algorithm to be used to compute the tour:\n Options are:\n" +\
                            "  (e)    Exact \n"                                                  +\
                            "  (t)    TSP   \n"                                                  +\
                            "  (tl)   TSP   (using approximate L1 ordering)\n"                   +\
                            "  (k)    k2-center   \n"                                            +\
                            "  (kl)   k2-center (using approximate L1 ordering)\n"               +\
                            "  (g)    Greedy\n"                                                  +\
                            "  (gl)   Greedy (using approximate L1 ordering])\n"                 +\
                            "  (gincex) Greedy Incremental(exact post optimization with slsqp)\n"                                      +\
                            "  (gincoll) Greedy Incremental(no post optimization, just colinear)\n"                                      +\
                            "  (gincl1) Greedy Incremental(using approximate L1 ordering)\n"                                      +\
                            "  (phi-mst) Compute the phi-prim-mst "                              +\
                            Style.RESET_ALL)

                    input_str = input_str.lstrip()

                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   input_str == 'e':
                          horseflytour = \
                                 run.getTour( algo_dumb,
                                              phi )
                    elif input_str == 'k': 
                          horseflytour = \
                                 run.getTour( algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=algo_exact_given_specific_ordering)
                          print " "
                          print Fore.GREEN, horseflytour['tour_points'], Style.RESET_ALL
                    elif input_str == 'kl':
                          horseflytour = \
                                 run.getTour( algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=algo_approximate_L1_given_specific_ordering)
                    elif input_str == 't':
                          horseflytour = \
                                 run.getTour( algo_tsp_ordering,
                                              phi,
                                              post_optimizer=algo_exact_given_specific_ordering)
                    elif input_str == 'tl':
                          horseflytour = \
                                 run.getTour( algo_tsp_ordering,
                                              phi,
                                              post_optimizer= algo_approximate_L1_given_specific_ordering)
                    elif input_str == 'g':
                          horseflytour = \
                                 run.getTour( algo_greedy,
                                              phi,
                                              post_optimizer= algo_exact_given_specific_ordering)
                    elif input_str == 'gl':
                          horseflytour = \
                                 run.getTour( algo_greedy,
                                              phi,
                                              post_optimizer= algo_approximate_L1_given_specific_ordering)
                                              
                    elif input_str == 'gincex':
                          horseflytour = \
                                 run.getTour( algo_greedy_incremental_insertion,
                                              phi, post_optimizer= algo_exact_given_specific_ordering)

                    elif input_str == 'gincoll':
                          horseflytour = \
                                 run.getTour( algo_greedy_incremental_insertion,
                                              phi, post_optimizer=None)

                    elif input_str == 'gincl1':
                          horseflytour = \
                                 run.getTour( algo_greedy_incremental_insertion,
                                              phi, post_optimizer=algo_approximate_L1_given_specific_ordering)


                    elif input_str == 'phi-mst':
                          phi_mst = \
                                 run.computeStructure(compute_phi_prim_mst ,phi)     
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    #print horseflytour['tour_points']

                    if input_str not in ['phi-mst']:
                         plotTour(horseflytour, run.inithorseposn, phi, input_str)
                    elif input_str == 'phi-mst':
                         draw_phi_mst(ax, phi_mst, run.inithorseposn, phi)
                         
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
    run = HorseFlyInput()
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
    

# Local data-structures for classic horsefly
class HorseFlyInput:
      def __init__(self, sites=[], inithorseposn=()):
           self.sites         = sites
           self.inithorseposn = inithorseposn

      # Methods for \verb|HorseFlyInput|
      def clearAllStates (self):
         self.sites = []
         self.inithorseposn = ()

      def getTour(self, algo, speedratio, k=None, post_optimizer=None):
       
          if k==None and post_optimizer==None:
                return algo(self.sites, self.inithorseposn, speedratio)
          elif k == None:
                return algo(self.sites, self.inithorseposn, speedratio, post_optimizer=post_optimizer)
          else:
                return algo(self.sites, self.inithorseposn, speedratio, k, post_optimizer=post_optimizer)
      def  computeStructure(self, structure_func, phi):
         print Fore.RED, "Computing the phi-mst", Style.RESET_ALL
         return structure_func(self.sites, self.inithorseposn, phi)
      def __repr__(self):

        if self.sites != []:
           tmp = ''
           for site in self.sites:
               tmp = tmp + '\n' + str(site)
           sites = "The list of sites to be serviced are " + tmp    
        else:
           sites = "The list of sites is empty"

        if self.inithorseposn != ():
           inithorseposn = "\nThe initial position of the horse is " + str(self.inithorseposn)
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
   
def algo_dumb(sites, horseflyinit, phi):
      
    tour_length_fn = tour_length(horseflyinit)
    best_tour      = algo_exact_given_specific_ordering(sites, horseflyinit, phi)
    i              = 0

    for sites_perm in list(itertools.permutations(sites)):

        print "Testing a new permutation ", i, " of the sites"; i = i + 1
        tour_for_current_perm = algo_exact_given_specific_ordering (sites_perm, horseflyinit, phi) 

        if tour_length_fn(utils_algo.flatten_list_of_lists(tour_for_current_perm ['tour_points']) ) \
         < tour_length_fn(utils_algo.flatten_list_of_lists(            best_tour ['tour_points']) ):

                best_tour = tour_for_current_perm
                print Fore.RED + "Found better tour!" + Style.RESET_ALL

    #print Fore.RED + "\nHorse Waiting times are ", best_tour['horse_waiting_times'] , Style.RESET_ALL
    return best_tour
   
def algo_greedy(sites, inithorseposn, phi, 
                write_algo_states_to_disk_p = True   ,
                animate_schedule_p          = True   , 
                post_optimizer              = None):

      # Set log, algo-state and input-output files config for \verb|algo_greedy|
        
      import sys, logging, datetime, os, errno

      algo_name     = 'algo-greedy-nearest-neighbor'
      time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
      dir_name      = algo_name + '---' + time_stamp
      log_file_name = dir_name + '/' + 'run.log'
      io_file_name  = 'input_and_output.yml'

      # Create directory for writing data-files and logs to for 
      # current run of this algorithm
      try:
          os.makedirs(dir_name)
      except OSError as e:
          if e.errno != errno.EEXIST:
              raise

      logging.basicConfig( filename = log_file_name,
                           level    = logging.DEBUG,
                           format   = '%(asctime)s: %(levelname)s: %(message)s',
                           filemode = 'w' )
      #logger = logging.getLogger()
      info("Started running greedy_nearest_neighbor for classic horsefly")

      algo_state_counter = 0 
      
      # Define function \verb|next_rendezvous_point_for_horse_and_fly|
         
      def next_rendezvous_point_for_horse_and_fly(horseposn, site):

           horseflytour = algo_exact_given_specific_ordering([site], horseposn, phi)
           return horseflytour['tour_points'][-1]
      
      # Define function \verb|greedy|
         
      def greedy(current_horse_posn, remaining_sites):

          if len(remaining_sites) == 1:
                return remaining_sites
          else:
                from scipy import spatial
                tree          = spatial.KDTree(remaining_sites)
                pts           = np.array([current_horse_posn])
                query_result  = tree.query(pts)
                next_site_idx = query_result[1][0]
                next_site     = remaining_sites[next_site_idx]

                next_horse_posn = next_rendezvous_point_for_horse_and_fly(current_horse_posn, next_site)
                remaining_sites.pop(next_site_idx) # the pop method modifies the list in place. 
                       
                return [next_site] + greedy(current_horse_posn = next_horse_posn, remaining_sites = remaining_sites)
      

      sites1                  = sites[:]
      sites_ordered_by_greedy = greedy(inithorseposn, remaining_sites=sites1)
      answer                  = post_optimizer(sites_ordered_by_greedy, inithorseposn, phi)
    
      # Write input and output of \verb|algo_greedy| to file
      
      data = {'visited_sites'  : answer['site_ordering'] ,
              'horse_tour'     : [inithorseposn] + answer['tour_points']   , 
              'phi'            : phi                     , 
              'inithorseposn'  : inithorseposn}

      import yaml
      with open(dir_name + '/' + io_file_name, 'w') as outfile:
           yaml.dump( data, \
                      outfile, \
                      default_flow_style=False)
      debug("Dumped input and output to " + io_file_name)
      
      # Make an animation of the schedule computed by \verb|algo_greedy|, if \verb|animate_schedule_p == True|
      
      if animate_schedule_p : 
           animateSchedule(dir_name + '/' + io_file_name)
      
      return answer
def algo_exact_given_specific_ordering (sites, horseflyinit, phi):

    # Useful functions for \verb|algo_exact_given_specific_ordering|
    
    def ith_leg_constraint(i, horseflyinit, phi, sites):
            if i == 0 :
                def _constraint_function(x):
                
                    #print "Constraint  ", i
                    start = np.array (horseflyinit)
                    site  = np.array (sites[0])
                    stop  = np.array ([x[0],x[1]])
                
                    horsetime = np.linalg.norm( stop - start )
                
                    flytime_to_site   = 1/phi * np.linalg.norm( site - start )
                    flytime_from_site = 1/phi * np.linalg.norm( stop - site  )
                    flytime           = flytime_to_site + flytime_from_site
                    return horsetime-flytime

                return _constraint_function
            else :
              
                def _constraint_function(x):

                   #print "Constraint  ", i
                   start = np.array (  [x[2*i-2], x[2*i-1]]  ) 
                   site  = np.array (  sites[i])
                   stop  = np.array (  [x[2*i]  , x[2*i+1]]  )
                
                   horsetime = np.linalg.norm( stop - start )
               
                   flytime_to_site   = 1/phi * np.linalg.norm( site - start )
                   flytime_from_site = 1/phi * np.linalg.norm( stop - site  )
                   flytime           = flytime_to_site + flytime_from_site
                   return horsetime-flytime

                return _constraint_function
    
    def generate_constraints(horseflyinit, phi, sites):
       cons = []
       for i in range(len(sites)):
            cons.append({'type':'eq','fun': ith_leg_constraint(i,horseflyinit,phi,sites)})
       return cons
    
    
    cons = generate_constraints(horseflyinit, phi, sites)
    
    # Initial guess for the non-linear solver.
    #x0 = np.empty(2*len(sites)); x0.fill(0.5) # choice of filling vector with 0.5 is arbitrary
    x0 = utils_algo.flatten_list_of_lists(sites) # the initial choice is just the sites

    assert(len(x0) == 2*len(sites))

    x0                  = np.array(x0)
    sol                 = minimize(tour_length(horseflyinit), x0, method= 'SLSQP', \
                                   constraints=cons         , options={'maxiter':500})
    tour_points         = utils_algo.pointify_vector(sol.x)
    numsites            = len(sites)
    alpha               = horseflyinit[0]
    beta                = horseflyinit[1]
    s                   = utils_algo.flatten_list_of_lists(sites)
    horse_waiting_times = np.zeros(numsites)
    ps                  = sol.x

    for i in range(numsites):

        if i == 0 :
            horse_time         = np.sqrt((ps[0]-alpha)**2 + (ps[1]-beta)**2)
            fly_time_to_site   = 1.0/phi * np.sqrt((s[0]-alpha)**2 + (s[1]-beta)**2 )
            fly_time_from_site = 1.0/phi * np.sqrt((s[0]-ps[1])**2 + (s[1]-ps[1])**2)
        else:
            horse_time         = np.sqrt((ps[2*i]-ps[2*i-2])**2 + (ps[2*i+1]-ps[2*i-1])**2)
            fly_time_to_site   = 1.0/phi * np.sqrt(( (s[2*i]-ps[2*i-2])**2 + (s[2*i+1]-ps[2*i-1])**2 )) 
            fly_time_from_site = 1.0/phi * np.sqrt(( (s[2*i]-ps[2*i])**2   + (s[2*i+1]-ps[2*i+1])**2 )) 

        horse_waiting_times[i] = horse_time - (fly_time_to_site + fly_time_from_site)
    
    return {'tour_points'                : tour_points,
            'horse_waiting_times'        : horse_waiting_times, 
            'site_ordering'              : sites,
            'tour_length_with_waiting_time_included': \
                                       tour_length_with_waiting_time_included(\
                                                    tour_points, \
                                                    horse_waiting_times, 
                                                    horseflyinit)}
 
def  algo_approximate_L1_given_specific_ordering(sites, horseflyinit, phi):
    import mosek
    numsites = len(sites)

    def p(idx):
        return idx + 0*numsites

    def b(idx):
        return idx + 2*numsites

    def f(idx):
        return idx + 4*numsites

    def h(idx):
        return idx + 6*numsites
    
    # Define a stream printer to grab output from MOSEK
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    numcon = 9 + 13*(numsites-1) # the first site has 9 constraints while the remaining n-1 sites have 13 constraints each
    numvar = 8 * numsites # Each ``L1 triangle'' has 8 variables associated with it

    alpha = horseflyinit[0]
    beta  = horseflyinit[1]

    s = utils_algo.flatten_list_of_lists(sites)

    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)
            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)
            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for idx in range(numvar):
                if (0   <= idx) and (idx < 2*numsites): # free variables (p section of the vector)
                    task.putvarbound(idx, mosek.boundkey.fr, -np.inf, np.inf)
                    
                elif  idx == 2*numsites : # b_0 is a known variable
                    val = abs(s[0]-alpha)
                    task.putvarbound(idx, mosek.boundkey.fx, val, val)
                
                elif  idx == 2*numsites +1 : # b_1 is a known variable
                    val = abs(s[1]-beta)
                    task.putvarbound(idx, mosek.boundkey.fx, val, val)

                else : # b_2, onwards and the f and h sections of the vector
                    task.putvarbound(idx, mosek.boundkey.lo, 0.0, np.inf)
                    
            # All the coefficients corresponding to the h's are 1's
            # and for the others the coefficients are 0. 
            for i in range(numvar):
                if i >= 6*numsites: # the h-section
                    task.putcj(i,1)
                else: # the p,b,f sections of x
                    task.putcj(i,0)

            # Constraints for the zeroth triangle corresponding to the zeroth site
            row = -1
            row += 1; task.putconbound(row, mosek.boundkey.up, -np.inf, alpha ) ; task.putarow(row, [p(0), h(0)],[1.0, -1.0])
            row += 1; task.putconbound(row, mosek.boundkey.lo, alpha  , np.inf) ; task.putarow(row, [p(0), h(0)],[1.0,  1.0])

            row += 1; task.putconbound(row, mosek.boundkey.up, -np.inf, beta ) ; task.putarow(row, [p(1), h(1)],[1.0, -1.0])
            row += 1; task.putconbound(row, mosek.boundkey.lo, beta  , np.inf) ; task.putarow(row, [p(1), h(1)],[1.0,  1.0])
            
            row += 1; task.putconbound(row, mosek.boundkey.up, -np.inf, s[0]  ) ; task.putarow(row, [p(0), f(0)],[1.0, -1.0])
            row += 1; task.putconbound(row, mosek.boundkey.lo,  s[0]  , np.inf) ; task.putarow(row, [p(0), f(0)],[1.0,  1.0])

            row += 1; task.putconbound(row, mosek.boundkey.up, -np.inf, s[1]  ) ; task.putarow(row, [p(1), f(1)],[1.0, -1.0])
            row += 1; task.putconbound(row, mosek.boundkey.lo,  s[1]  , np.inf) ; task.putarow(row, [p(1), f(1)],[1.0,  1.0])

            # The most important constraint of all! On the ``L1 triangle''
            # time for drone to start from the truck reach site and get back to truck
            # = time for truck between the two successive rendezvous points
            # The way I have modelled the following constraint it is not exactly
            # the same as the previous statement of equality of times of truck
            # and drone, but for initial experiments it looks like this gives
            # waiting times to be automatically close to 0 (1e-9 close to machine-epsilon)
            # Theorem in the making?? 
            row += 1; task.putconbound(row, mosek.boundkey.fx, 0.0 , 0.0 ) ;
            task.putarow(row, [b(0), b(1), f(0), f(1), h(0), h(1)], [1.0,1.0,1.0,1.0,-phi, -phi])

            # Constraints beginning from the 1st triangle
            for  i in range(1,numsites):
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, -s[2*i]  , np.inf) ; task.putarow(row, [b(2*i),   p(2*i-2)],[1.0, -1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo,  s[2*i]  , np.inf) ; task.putarow(row, [b(2*i),   p(2*i-2)],[1.0,  1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, -s[2*i+1], np.inf) ; task.putarow(row, [b(2*i+1), p(2*i-1)],[1.0, -1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo,  s[2*i+1], np.inf) ; task.putarow(row, [b(2*i+1), p(2*i-1)],[1.0,  1.0])
                
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, -s[2*i]  , np.inf) ; task.putarow(row, [f(2*i),    p(2*i)]  ,[1.0, -1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo,  s[2*i]  , np.inf) ; task.putarow(row, [f(2*i),    p(2*i)]  ,[1.0,  1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, -s[2*i+1], np.inf) ; task.putarow(row, [f(2*i+1),  p(2*i+1)],[1.0, -1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo,  s[2*i+1], np.inf) ; task.putarow(row, [f(2*i+1),  p(2*i+1)],[1.0,  1.0])
                
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, 0.0     , np.inf); task.putarow(row, [p(2*i)  , p(2*i-2), h(2*i)]  , [1.0,-1.0, 1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.up, -np.inf , 0.0   ); task.putarow(row, [p(2*i)  , p(2*i-2), h(2*i)]  , [1.0,-1.0,-1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.lo, 0.0     , np.inf); task.putarow(row, [p(2*i+1), p(2*i-1), h(2*i+1)], [1.0,-1.0, 1.0])
                row+=1 ;  task.putconbound(row, mosek.boundkey.up, -np.inf , 0.0   ); task.putarow(row, [p(2*i+1), p(2*i-1), h(2*i+1)], [1.0,-1.0,-1.0])
                # The most important constraint of all! On the ``L1 triangle''
                # time for drone to start from the truck reach site and get back to truck
                # = time for truck between the two successive rendezvous points
                row+=1; task.putconbound(row, mosek.boundkey.fx, 0.0 , 0.0 ) ;
                task.putarow(row, [b(2*i), b(2*i+1), f(2*i), f(2*i+1), h(2*i), h(2*i+1)], [1.0,1.0,1.0,1.0,-phi, -phi])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            #task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.bas)
            
            if (solsta == mosek.solsta.optimal or
                        solsta == mosek.solsta.near_optimal):
                    xx = [0.] * numvar
                        # Request the basic solution.
                    task.getxx(mosek.soltype.bas, xx)
                    #print("Optimal solution: ")
                    #for i in range(numvar):
                    #    print("x[" + str(i) + "]=" + str(xx[i]))
            elif (solsta == mosek.solsta.dual_infeas_cer or
                    solsta == mosek.solsta.prim_infeas_cer or
                    solsta == mosek.solsta.near_dual_infeas_cer or
                    solsta == mosek.solsta.near_prim_infeas_cer):
                    print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                    print("Unknown solution status")
            else:
                    print("Other solution status")

            # Now that we have solved the LP
            # We need to extract the ``p'' section of the vector
            ps = xx[:2*numsites]
            bs = xx[2*numsites:4*numsites]
            fs = xx[4*numsites:6*numsites]
            hs = xx[6*numsites:]

            ######################################################################################
            # This commented out section is important to check how close to zero the waiting times
            # are as calculated by the LP. To understand this, comment in this section and comment
            # out the part using tghe L2 metric below it
            ######################################################################################
            # horse_waiting_times = np.zeros(numsites)
            # for i in range(numsites):
            #     if i == 0 :
            #         horse_time         = abs(ps[0]-alpha) + abs(ps[1]-beta)
            #         fly_time_to_site   = 1.0/phi * (abs(s[0]-alpha) + abs(s[1]-beta))
            #         fly_time_from_site = 1.0/phi * (abs(s[0]-ps[1]) + abs(s[1]-ps[1]))
            #     else:
            #         horse_time         = abs(ps[2*i]-ps[2*i-2]) + abs(ps[2*i+1]-ps[2*i-1])
            #         fly_time_to_site   = 1.0/phi * ( abs(s[2*i]-ps[2*i-2]) + abs(s[2*i+1]-ps[2*i-1]) ) 
            #         fly_time_from_site = 1.0/phi * ( abs(s[2*i]-ps[2*i])   + abs(s[2*i+1]-ps[2*i+1]) ) 
            #     horse_waiting_times[i] = horse_time - (fly_time_to_site + fly_time_from_site)

            horse_waiting_times = np.zeros(numsites)
            for i in range(numsites):
                if i == 0 :
                    horse_time         = np.sqrt((ps[0]-alpha)**2 + (ps[1]-beta)**2)
                    fly_time_to_site   = 1.0/phi * np.sqrt((s[0]-alpha)**2 + (s[1]-beta)**2)
                    fly_time_from_site = 1.0/phi * np.sqrt((s[0]-ps[1])**2 + (s[1]-ps[1])**2)
                else:
                    horse_time         = np.sqrt((ps[2*i]-ps[2*i-2])**2 + (ps[2*i+1]-ps[2*i-1])**2)
                    fly_time_to_site   = 1.0/phi * np.sqrt( (s[2*i]-ps[2*i-2])**2 + (s[2*i+1]-ps[2*i-1])**2 ) 
                    fly_time_from_site = 1.0/phi * np.sqrt( (s[2*i]-ps[2*i])**2   + (s[2*i+1]-ps[2*i+1])**2 ) 
                    
                horse_waiting_times[i] = horse_time - (fly_time_to_site + fly_time_from_site)
                
            tour_points = utils_algo.pointify_vector(ps)
            return {'tour_points'        : tour_points,
                    'horse_waiting_times': horse_waiting_times, 
                    'site_ordering'      : sites,
                    'tour_length_with_waiting_time_included': tour_length_with_waiting_time_included(tour_points, horse_waiting_times, horseflyinit)}

 

# Define auxiliary helper functions
def single_site_solution(site, horseposn, phi):
     
     h = np.asarray(horseposn)
     s = np.asarray(site)
     
     hs_mag  = 1.0/np.linalg.norm(s-h) 
     hs_unit = 1.0/hs_mag * (s-h)
     
     r      = h +  2*hs_mag/(1+phi) * hs_unit # Rendezvous point
     hr_mag = np.linalg.norm(r-h)

     return (tuple(r), hr_mag) 
def compute_collinear_horseflies_tour_length(sites, horseposn, phi):

     if not sites: # No more sites, left to visit!
          return 0
     else:         # Some sites are still left on the itinerary

          (rendezvous_pt, horse_travel_length) = single_site_solution(sites[0], horseposn, phi )
          return horse_travel_length  + \
                 compute_collinear_horseflies_tour_length( sites[1:], rendezvous_pt, phi )
def compute_collinear_horseflies_tour(sites, inithorseposn, phi):

      horseposn         = inithorseposn
      horse_tour_points = [inithorseposn]

      for site in sites:
          (rendezvous_pt, _) = single_site_solution(site, horseposn, phi )
            
          horse_tour_points.append(rendezvous_pt)
          horseposn = rendezvous_pt

      return horse_tour_points

# Define various insertion policy classes
class PolicyBestInsertionNaive:

    def __init__(self, sites, inithorseposn, phi):

         self.sites           = sites
         self.inithorseposn   = inithorseposn
         self.phi             = phi

         self.visited_sites        = []                # The actual list of visited sites (not indices)
         self.unvisited_sites_idxs = range(len(sites)) # This indexes into self.sites
         self.horse_tour           = [self.inithorseposn]         

    # Methods for \verb|PolicyBestInsertionNaive|
    def insert_another_unvisited_site(self):
       # Compute the length of the tour that currently services the visited sites
       current_tour_length    = \
                compute_collinear_horseflies_tour_length(\
                           self.visited_sites,\
                           self.inithorseposn,\
                           self.phi) 
           
       delta_increase_least_table = [] # tracking variable updated in for loop below

       for u in self.unvisited_sites_idxs:
          # Set up tracking variables local to this iteration
          ibest                = 0
          delta_increase_least = float("inf")
          
          # If \texttt{self.sites[u]} is chosen for insertion, find best insertion position and update \texttt{delta\_increase\_least\_table}
          for i in range(len(self.sites)):
                              
                      visited_sites_test = self.visited_sites[:i] +\
                                           [ self.sites[u] ]      +\
                                           self.visited_sites[i:]
                                                
                      tour_length_on_insertion = \
                                 compute_collinear_horseflies_tour_length(\
                                            visited_sites_test,\
                                            self.inithorseposn,\
                                            self.phi) 

                      delta_increase = tour_length_on_insertion - current_tour_length                         
                      assert(delta_increase >= 0)               

                      if delta_increase < delta_increase_least:
                            delta_increase_least = delta_increase
                            ibest                = i                                              
                                
          delta_increase_least_table.append({'unvisited_site_idx'      : u    , \
                                             'best_insertion_position' : ibest, \
                                             'delta_increase'          : delta_increase_least})
            
                     
       # Find the unvisited site which on insertion increases tour-length by the least amount
       best_table_entry = min(delta_increase_least_table, \
                                key = lambda x: x['delta_increase'])
                
       unvisited_site_idx_for_insertion = best_table_entry['unvisited_site_idx']
       insertion_position               = best_table_entry['best_insertion_position']
       delta_increase                   = best_table_entry['delta_increase']
            
       # Update states for \texttt{PolicyBestInsertionNaive}
       # Update visited and univisted sites info
       self.visited_sites = self.visited_sites[:insertion_position]      +\
                            [self.sites[unvisited_site_idx_for_insertion]] +\
                            self.visited_sites[insertion_position:]
         
       self.unvisited_sites_idxs = filter( lambda elt: elt != unvisited_site_idx_for_insertion, \
                                           self.unvisited_sites_idxs ) 

       # Update the tour of the horse
       self.horse_tour = compute_collinear_horseflies_tour(\
                                  self.visited_sites,         \
                                  self.inithorseposn, \
                                  self.phi) 
        
    

def algo_greedy_incremental_insertion(sites, inithorseposn, phi,
                                      insertion_policy_name       = "naive",
                                      write_algo_states_to_disk_p = False  ,
                                      animate_schedule_p          = True   , 
                                      post_optimizer              = None   ,  
                                      plot_computed_schedule      = False):
      # Set log, algo-state and input-output files config
        
      import sys, logging, datetime, os, errno

      algo_name     = 'algo-greedy-incremental-insertion'
      time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
      dir_name      = algo_name + '---' + time_stamp
      log_file_name = dir_name + '/' + 'run.log'
      io_file_name  = 'input_and_output.yml'

      # Create directory for writing data-files and logs to for 
      # current run of this algorithm
      try:
          os.makedirs(dir_name)
      except OSError as e:
          if e.errno != errno.EEXIST:
              raise

      logging.basicConfig( filename = log_file_name,
                           level    = logging.DEBUG,
                           format   = '%(asctime)s: %(levelname)s: %(message)s',
                           filemode = 'w' )
      #logger = logging.getLogger()
      info("Started running greedy_incremental_insertion for classic horsefly")

      algo_state_counter = 1 
      
      # Set insertion policy class for current run
      
      if insertion_policy_name == "naive":
           insertion_policy = PolicyBestInsertionNaive(sites, inithorseposn, phi)
      else: 
           print insertion_policy_name
           sys.exit("Unknown insertion policy: ")
      debug("Finished setting insertion policy: " + insertion_policy_name)
      

      while insertion_policy.unvisited_sites_idxs: 
         # Use insertion policy to find the cheapest site to insert into current tour
         insertion_policy.insert_another_unvisited_site()
         debug(Fore.GREEN + "Inserted another unvisited site" + Style.RESET_ALL)
         
         # Write algorithms current state to file
         if write_algo_states_to_disk_p:
              import yaml
              algo_state_file_name = 'algo_state_'                    + \
                                str(algo_state_counter).zfill(5) + \
                                '.yml'

              data = {'insertion_policy_name' : insertion_policy_name                       ,
                      'unvisited_sites'       : [insertion_policy.sites[u] \
                                                     for u in insertion_policy.unvisited_sites_idxs], 
                      'visited_sites'         : insertion_policy.visited_sites                    , 
                      'horse_tour'            : insertion_policy.horse_tour }

              with open(dir_name + '/' + algo_state_file_name, 'w') as outfile:
                   yaml.dump( data   , \
                              outfile, \
                              default_flow_style = False)
                   # Render current algorithm state to image file
                   import utils_algo
                   if write_algo_states_to_disk_p:
                        # Set up plotting area and canvas, fig, ax, and other configs
                        from matplotlib import rc
                        rc('font', **{'family': 'serif', \
                                   'serif': ['Computer Modern']})
                        rc('text', usetex=True)
                        fig,ax = plt.subplots()
                        ax.set_xlim([0,1])
                        ax.set_ylim([0,1])
                        ax.set_aspect(1.0)
                        ax = fig.gca()
                        ax.set_xticks(np.arange(0, 1, 0.1))     
                        ax.set_yticks(np.arange(0, 1., 0.1))
                        plt.grid(linestyle='dotted')
                        ax.set_xticklabels([]) # to remove those numbers at the bottom
                        ax.set_yticklabels([])

                        ax.tick_params(
                            bottom=False,      # ticks along the bottom edge are off
                            left=False,        # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
                          
                        # Extract $x$ and $y$ coordinates of the points on the horse, fly tours, visited and unvisited sites
                        # Route for the horse
                        xhs = [ data['horse_tour'][i][0] \
                                  for i in range(len(data['horse_tour']))  ]    
                        yhs = [ data['horse_tour'][i][1] \
                                  for i in range(len(data['horse_tour']))  ]    

                        # Route for the fly. The fly keeps alternating between the site and the horse
                        xfs , yfs = [xhs[0]], [yhs[0]]
                        for site, pt in zip (data['visited_sites'],
                                             data['horse_tour'][1:]):
                            xfs.extend([site[0], pt[0]])
                            yfs.extend([site[1], pt[1]])
                                
                        xvisited = [ data['visited_sites'][i][0] \
                                       for i in range(len(data['visited_sites']))  ]    
                        yvisited = [ data['visited_sites'][i][1] \
                                       for i in range(len(data['visited_sites']))  ]    
                            
                        xunvisited = [ data['unvisited_sites'][i][0] \
                                         for i in range(len(data['unvisited_sites']))  ]    
                        yunvisited = [ data['unvisited_sites'][i][1] 
                                         for i in range(len(data['unvisited_sites'])) ]    
                        debug("Extracted x and y coordinates for route of horse, fly, visited and unvisited sites")
                          
                        # Mark initial position of horse and fly boldly on canvas
                        ax.add_patch( mpl.patches.Circle( inithorseposn, \
                                                          radius = 1/55.0,\
                                                          facecolor= '#D13131', #'red',\
                                                          edgecolor='black')  )
                        debug("Marked the initial position of horse and fly on canvas")
                          
                        # Place numbered markers on visited sites to mark the order of visitation explicitly
                        for x,y,i in zip(xvisited, yvisited, range(len(xvisited))):
                             ax.text(x, y, str(i+1),  fontsize=8, \
                                     bbox=dict(facecolor='#ddcba0', alpha=1.0, pad=2.0)) 
                        debug("Placed numbered markers on visited sites")
                        
                        # Draw horse and fly-tours
                        ax.plot(xfs,yfs,'g-',linewidth=1.1)  
                        ax.plot(xhs, yhs, color='r', \
                                marker='s', markersize=3, \
                                linewidth=1.6) 
                        debug("Plotted the horse and fly tours")
                        
                        # Draw unvisited sites as filled blue circles
                        for x, y in zip(xunvisited, yunvisited):
                             ax.add_patch( mpl.patches.Circle( (x,y),\
                                                            radius    = 1/100.0,\
                                                            facecolor = 'blue',\
                                                            edgecolor = 'black')  )
                        debug("Drew univisted sites")
                        
                        # Give metainformation about current picture as headers and footers
                        fontsize = 15
                        ax.set_title( r'Number of sites visited so far: ' +\
                                       str(len(data['visited_sites']))   +\
                                       '/' + str(len(sites))           ,  \
                                            fontdict={'fontsize':fontsize})
                        ax.set_xlabel(r'$\varphi=$'+str(phi), fontdict={'fontsize':fontsize})
                        debug("Setting title, headers, footers, etc...")
                        
                        # Write image file
                        image_file_name = 'algo_state_'                    +\
                                          str(algo_state_counter).zfill(5) +\
                                             '.png'
                        plt.savefig(dir_name + '/' + image_file_name,  \
                                    bbox_inches='tight', dpi=250)
                        print "Wrote " + image_file_name + " to disk"   
                        plt.close() 
                        debug(Fore.BLUE+"Rendered algorithm state to image file"+Style.RESET_ALL)
                        
                   

              algo_state_counter = algo_state_counter + 1
              debug("Dumped algorithm state to " + algo_state_file_name)
         

      # Run post optimizer on obtained tour
      
      if not (post_optimizer is None):
          import utils_algo
          print insertion_policy.horse_tour
          answer=post_optimizer(insertion_policy.visited_sites, inithorseposn, phi)
          insertion_policy.horse_tour = [inithorseposn] + answer['tour_points']
          print "  "
          print insertion_policy.horse_tour
          #sys.exit()
      
      # Write input and output to file
      # ASSERT: `inithorseposn` is included as first point of the tour
      assert(len(insertion_policy.horse_tour) == len(insertion_policy.visited_sites) + 1) 

      # ASSERT: All sites have been visited. Simple sanity check 
      assert(len(insertion_policy.sites)   == len(insertion_policy.visited_sites)) 

      data = {'insertion_policy_name' : insertion_policy_name   ,
              'visited_sites'  : insertion_policy.visited_sites , 
              'horse_tour'     : insertion_policy.horse_tour    , 
              'phi'            : insertion_policy.phi           , 
              'inithorseposn'  : insertion_policy.inithorseposn}

      import yaml
      with open(dir_name + '/' + io_file_name, 'w') as outfile:
           yaml.dump( data, \
                      outfile, \
                      default_flow_style=False)
      debug("Dumped input and output to " + io_file_name)
      
      # Make an animation of the schedule, if \verb|animate_schedule_p == True|
         
      if animate_schedule_p : 
           animateSchedule(dir_name + '/' + io_file_name)
      
      #sys.exit()
      # Make an animation of algorithm states, if \verb|write_algo_states_to_disk_p == True|
      if write_algo_states_to_disk_p:
           import subprocess, os
           os.chdir(dir_name)
           subprocess.call( ['ffmpeg',  '-hide_banner', '-loglevel', 'verbose', \
                             '-r', '1',  '-i', 'algo_state_%05d.png', \
                             '-vcodec', 'mpeg4', '-r', '10' , \
                             'algo_state_animation.avi']  )
           os.chdir('../')
      
      # Return horsefly tour, along with additional information
      debug("Returning answer")
      horse_waiting_times = np.zeros(len(sites)) # TODO write this to file later
      return {'tour_points'                : insertion_policy.horse_tour[1:],
              'horse_waiting_times'        : horse_waiting_times, 
              'site_ordering'              : insertion_policy.visited_sites,
              'tour_length_with_waiting_time_included': \
                                             tour_length_with_waiting_time_included(\
                                                          insertion_policy.horse_tour[1:], \
                                                          horse_waiting_times, \
                                                          inithorseposn)}
      
   
def algo_kmeans(sites, inithorseposn, phi, k, post_optimizer):
     """
     type Point   (Double, Double)
     type Site    Point
     type Cluster (Point, [Site])
     type Tour    {'site_ordering':[Site], 
                   'tour_points'  :[Point]}
     algo_kmeans :: [Site] -> Point -> Double -> Int
     """
     def get_clusters(site_list):
           """ 
           get_clusters :: [Site] -> [Cluster]
           For the given list of sites, perform k-means clustering
           and return the list of k-centers, along with a list of sites
           assigned to each center. 
           """
           X      = np.array(site_list)
           kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

           accum = [ (center, [])  for center in kmeans.cluster_centers_ ]
           for label, site in zip(kmeans.labels_, site_list):
                 accum [label][1].append(site)

           return accum 

     def extract_cluster_sites_for_each_cluster(clusters):
         """
         extract_cluster_sites_for_each_cluster :: [Cluster] -> [[Site]]
         """
         return [ cluster_sites for (_, cluster_sites) in clusters  ]

     def fuse_tours(tours):
         """ 
          fuse_tours :: [Tour] -> Tour
         """
         fused_tour = {'site_ordering':[], 'tour_points':[]}
         for tour, i in zip(tours, range(len(tours))):
               fused_tour['site_ordering'].extend(tour['site_ordering'])
               if i != len(tours)-1:
                     # Remember! last point of previous tour is first point of
                     # this tour, which is why we need to avoid duplication
                     # Hence the [:-1]
                     fused_tour['tour_points'].extend(tour['tour_points'][:-1]) 
               else: 
                     # Because this is the last tour in the iteration, we include
                     # its end point also, hence no [:-1] here
                     fused_tour['tour_points'].extend(tour['tour_points'])
         return fused_tour

     def  weighted_center_tour(clusters, horseflyinit):
         """ 
         weighted_center_tour :: [Cluster] -> Point -> [Cluster]
         
         Just return a permutation of the clusters. 
         need to return actual weighted tour
         since we are only interested in the order
         in which the weighted center tour is performed
         on k weighted points, where k is the clustering 
         number used here
         """
         
         #print Fore.CYAN, " Clusters: "    , clusters, Style.RESET_ALL
         #print " "
         #print Fore.CYAN, " Horseflyinit: ", horseflyinit, Style.RESET_ALL
         
         assert( k == len(clusters) )
         tour_length_fn = tour_length(horseflyinit)

         #-------------------------------------------------
         # For each of the k! permutations of the weighted sites
         # give the permutation with the smallest weighted tour
         # Note that k is typically small, say 2,3 or 4
         #-------------------------------------------------
         # But first we initialize the accumulator variables prefixed with best_

         #print Fore.YELLOW , " Computing Weighted Center Tour ", Style.RESET_ALL
         clustering_centers = [ center          for (center, _)    in clusters]
         centers_weights   =  [ len(site_list)  for (_, site_list) in clusters]

         #utils_algo.print_list(clustering_centers)
         #utils_algo.print_list(centers_weights)
         #time.sleep(5000)

         best_perm = clusters
         best_perm_tour = algo_weighted_sites_given_specific_ordering(clustering_centers, \
                                                               centers_weights, \
                                                               horseflyinit, \
                                                               phi)

         i = 1
         for clusters_perm in list(itertools.permutations(clusters)):

               #print Fore.YELLOW , "......Testing a new cluster permutation [ ", i ,  \
               #                     "/", math.factorial(k) , " ] of the sites", \
               #                    Style.RESET_ALL

               i = i + 1
               # cluster_centers_and_weights ::  [(Point, Int)]
               # This is what is used for the weighted tour
               clustering_centers = [ center          for (center, _)    in clusters_perm]
               centers_weights    = [ len(site_list)  for (_, site_list) in clusters_perm] 
               
               tour_current_perm = \
                   algo_weighted_sites_given_specific_ordering(clustering_centers, \
                                                               centers_weights, \
                                                               horseflyinit, \
                                                               phi)

               if tour_length_fn( utils_algo.flatten_list_of_lists(tour_current_perm ['tour_points']) ) \
                < tour_length_fn( utils_algo.flatten_list_of_lists(   best_perm_tour ['tour_points']) ):
 
                   print Fore.RED + ".................Found better cluster order" + Style.RESET_ALL
                   best_perm = clusters_perm

         return best_perm
               
     def get_tour (site_list, horseflyinit):
        """
        get_tour :: [Site] -> Point -> Tour
        
        A recursive function which does the job 
        of extracting a tour
        """

        if len (site_list) <= k: # Base-case for the recursion
              #print Fore.CYAN + ".....Reached Recursion Base case" + Style.RESET_ALL
              result = algo_dumb(site_list, horseflyinit, phi)
              return result 
        else: # The main recursion
           # Perform k-means clustering and get the clusters
           clusters = get_clusters(site_list)

           #utils_algo.print_list(clusters)

           ###################################################################
           # Permute the clusters depending on which is better to visit first
           clusters_perm = weighted_center_tour(clusters, horseflyinit)
           ####################################################################

           # Extract cluster sites for each cluster
           cluster_sites_for_each_cluster  = \
                  extract_cluster_sites_for_each_cluster(clusters_perm)

           # Apply the get_tour function on each chunk while folding across
           # using the last point of the tour of the previous cluster
           # as the first point of this current one. This is a kind of recursion
           # that pays forward.
           tours = []
           for site_list, i in zip(cluster_sites_for_each_cluster,
                                   range(len(cluster_sites_for_each_cluster))):
                 
                 if i == 0:# first point is horseflyinit. The starting fold value!!
                       tours.append( get_tour(site_list, inithorseposn)  )
                 else: # use the last point of the previous tour (i-1 index)
                       # as the first point of this one !!
                       prev_tour  = tours[i-1]
                       tours.append( get_tour(site_list, prev_tour['tour_points'][-1]))
           # Fuse the tours you obtained above to get a site ordering
           return fuse_tours(tours)

     print Fore.MAGENTA + "RUNNING algo_kmeans......." + Style.RESET_ALL
     sites1 = get_tour(sites, inithorseposn)['site_ordering']
     return  post_optimizer(sites1, inithorseposn, phi )



def algo_weighted_sites_given_specific_ordering (sites, weights, horseflyinit, phi):
      
     def site_constraints(i, sites, weights):
          """
          site_constraints :: Int -> [Site] -> [Double] 
                          -> [ [Double] -> Double  ]
          
          Generate a list of constraint functions for the ith site
          The number of constraint functions is equal to the weight
          of the site!
          """

          #print Fore.RED, sites, Style.RESET_ALL
         
          psum_weights = utils_algo.partial_sums( weights ) # partial sum of ALL the site-weights
          accum        = [ ]
          site_weight  = weights[i]

          for j in range(site_weight): 

              if i == 0 and j == 0:

                    #print "i= ", i, " j= ", j
                    def _constraint_function(x):
                        """
                        constraint_function :: [Double] -> Double
                        """
                        start = np.array (horseflyinit)
                        site  = np.array (sites[0])
                        stop  = np.array ([x[0],x[1]])
                        
                        horsetime = np.linalg.norm( stop - start )
                        
                        flytime_to_site   = 1/phi * np.linalg.norm( site - start )
                        flytime_from_site = 1/phi * np.linalg.norm( stop - site  )
                        flytime           = flytime_to_site + flytime_from_site
                        return horsetime-flytime
                    
                    accum.append( _constraint_function )
                    
              elif  i == 0 and j != 0 :

                    #print "i= ", i, " j= ", j
                    def _constraint_function(x):
                          """
                          constraint_function :: [Double] -> Double
                          """
                          start = np.array( [x[2*j-2], x[2*j-1]] ) 
                          site  = np.array(sites[0])
                          stop  = np.array( [x[2*j]  , x[2*j+1]] )

                          horsetime = np.linalg.norm( stop - start )
                          
                          flytime_to_site   = 1/phi * np.linalg.norm( site - start )
                          flytime_from_site = 1/phi * np.linalg.norm( stop - site  )
                          flytime           = flytime_to_site + flytime_from_site
                          return horsetime-flytime

                    accum.append( _constraint_function )
              else:

                    #print "i= ", i, " j= ", j
                    def _constraint_function(x):
                          """
                          constraint_function :: [Double] -> Double
                          """
                          
                          offset = 2 * psum_weights[i-1]
                          
                          start  = np.array( [ x[offset + 2*j-2 ], x[offset + 2*j-1 ] ] ) 
                          site   = np.array(sites[i])
                          stop   = np.array( [ x[offset + 2*j ]  , x[offset + 2*j+1 ] ] )

                          horsetime = np.linalg.norm( stop - start )
                          
                          flytime_to_site   = 1/phi * np.linalg.norm( site - start )
                          flytime_from_site = 1/phi * np.linalg.norm( stop - site  )
                          flytime           = flytime_to_site + flytime_from_site
                          return horsetime-flytime

                    accum.append( _constraint_function )

          return accum 

     def generate_constraints(sites, weights):
         return [site_constraints(i, sites, weights) for i in range(len(sites))]

     #####
     #print weights
     #### For debugging
     weights = [1 for wt in weights]
     ####
     
     cons = utils_algo.flatten_list_of_lists (generate_constraints(sites, weights))
     cons1 = [  {'type':'eq', 'fun':f}  for f in cons]
     
     # Since the horsely tour lies inside the square,
     # the bounds for each coordinate is 0 and 1
     x0 = np.empty(2*sum(weights))
     x0.fill(0.5) # choice of filling vector with 0.5 is arbitrary

     # Run scipy's minimization solver
     sol = minimize(tour_length(horseflyinit), x0, method= 'SLSQP', constraints=cons1)
     tour_points = utils_algo.pointify_vector(sol.x)

     #print sol

     #time.sleep(5000)
     return {'tour_points'  : tour_points,
             'site_ordering': sites}

def algo_tsp_ordering(sites, inithorseposn, phi, post_optimizer):
    import tsp
    horseinit_and_sites = [inithorseposn] + sites

    _, tsp_idxs = tsp.tsp(horseinit_and_sites)

          
    # Get the position of the horse in tsp_idxss
    h = tsp_idxs.index(0) # 0 because the horse was placed first in the above vector

    if h != len(tsp_idxs)-1:
        idx_vec = tsp_idxs[h+1:] + tsp_idxs[:h]
    else:
        idx_vec = tsp_idxs[:h]

    # idx-1 because all the indexes of the sites were pushed forward
    # by 1 when we tacked on inithorseposn at the very beginning
    # of horseinit_and_sites, hence we auto-correct for that
    sites_tsp = [sites[idx-1] for idx in idx_vec]
    
    tour0    = post_optimizer (sites_tsp                , inithorseposn, phi) 
    tour1    = post_optimizer (list(reversed(sites_tsp)), inithorseposn, phi) 
    
    tour0_length = utils_algo.length_polygonal_chain([inithorseposn] + tour0['site_ordering'])
    tour1_length = utils_algo.length_polygonal_chain([inithorseposn] + tour1['site_ordering'])

    print Fore.RED, " TSP paths in either direction are ", tour0_length, " ", tour1_length, Style.RESET_ALL
    
    if tour0_length < tour1_length:
        print Fore.RED, "Selecting tour0 ", Style.RESET_ALL
        return tour0
    else:
        print Fore.RED, "Selecting tour1 ", Style.RESET_ALL
        return tour1

# Lower bounds for classic horsefly
def compute_phi_prim_mst(sites, inithorseposn,phi):

     import networkx as nx
     from sklearn.neighbors import NearestNeighbors
     
     # Create singleton graph, with node at \verb|inithorseposn|
     G = nx.Graph()
     G.add_node(0, mycoordinates=inithorseposn, joined_site_coords=[])
     

     unmarked_sites_idxs = range(len(sites))
     while unmarked_sites_idxs:
          node_site_info = []
          
          # For each node, find the closest site
             
          for nodeid, nodeval in G.nodes.data():
              current_node_coordinates = np.asarray(nodeval['mycoordinates'])
              distances_of_current_node_to_sites = []
                         
              # The following loop finds the nearest unmarked site. So far, I am 
              # using brute force for this, later, I will use sklearn.neighbors.
              for j in unmarked_sites_idxs:
                  site_coordinates = np.asarray(sites[j])
                  dist             =  np.linalg.norm( site_coordinates - current_node_coordinates )
                              
                  distances_of_current_node_to_sites.append( (j, dist) )

                  nearest_site_idx, distance_of_current_node_to_nearest_site = \
                                  min(distances_of_current_node_to_sites, key=lambda (_, d): d)

                  node_site_info.append((nodeid, \
                                            nearest_site_idx, \
                                            distance_of_current_node_to_nearest_site))
              
          # Find the node with the closest site, and generate the next node and edge for the $\varphi$-MST
          
          opt_node_idx,          \
          next_site_to_mark_idx, \
          distance_to_next_site_to_mark = min(node_site_info, key=lambda (h,k,d) : d)

          tmp = sites[next_site_to_mark_idx]
          G.nodes[opt_node_idx]['joined_site_coords'].append(  tmp   ) 
          (r, h) = single_site_solution(tmp, G.nodes[opt_node_idx]['mycoordinates'], phi) 
                    
          # Remember! indexing of nodes started at 0, thats why you set
          # numnodes to the index of the newly inserted node. 
          newnodeid = len(list(G.nodes))

          # joined_site_coords will be updated in the future iterations of while :
          G.add_node(newnodeid, mycoordinates=r, joined_site_coords=[]) 
            
          # insert the edge weight, will be useful later when 
          # computing sum total of all the edges.
          G.add_edge(opt_node_idx, newnodeid, weight=h ) 
          
          
          # Marking means removing from unmarked list :-D
          unmarked_sites_idxs.remove(next_site_to_mark_idx)
          
     utils_algo.print_list(G.nodes.data())
     utils_algo.print_list(G.edges.data())
     return G

# Plotting routines for classic horsefly
def plotTour(horseflytour, horseflyinit, phi, algo_str, tour_color='#d13131'):
   
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

    # Get x and y coordinates of the endpoints of segments on the horse-tour
       
    xhs, yhs = [horseflyinit[0]], [horseflyinit[1]]
    for pt in horseflytour['tour_points']:
        xhs.append(pt[0])
        yhs.append(pt[1])
    
    # Get x and y coordinates of the sites
       
    xsites, ysites = [], []
    for pt in horseflytour['site_ordering']:
        xsites.append(pt[0])
        ysites.append(pt[1])
    
    # Construct the fly-tour from the information about horse tour and sites
    
    xfs , yfs = [xhs[0]], [yhs[0]]
    for site, pt in zip (horseflytour['site_ordering'],
                         horseflytour['tour_points']):
       xfs.extend([site[0], pt[0]])
       yfs.extend([site[1], pt[1]])
    
    # Print information about the horse tour
       
    print "\n----------", "\nHorse Tour", "\n-----------"
    waiting_times = [0.0] + horseflytour['horse_waiting_times'].tolist() 
    #print waiting_times
    for pt, time in zip(zip(xhs,yhs), waiting_times) :
       print pt, Fore.GREEN, " ---> Horse Waited ", time, Style.RESET_ALL
    
    # Print information about the fly tour
       
    print "\n----------", "\nFly Tour", "\n----------"
    for item, i in zip(zip(xfs,yfs), range(len(xfs))):
       if i%2 == 0:
           print item
       else :
           print Fore.RED + str(item) + "----> Site" +  Style.RESET_ALL

    
    # Print meta-data about the algorithm run
       
    print "----------------------------------"
    print Fore.GREEN, "\nSpeed of the drone was set to be", phi
    #tour_length = utils_algo.length_polygonal_chain( zip(xhs, yhs))
    tour_length = horseflytour['tour_length_with_waiting_time_included']
    print "Tour length of the horse is ",  tour_length
    print "Algorithm code-Key used "    , algo_str, Style.RESET_ALL
    print "----------------------------------\n"
    
    # Plot everything
     
    #kwargs = {'size':'large'}
    for x,y,i in zip(xsites, ysites, range(len(xsites))):
        ax.text(x, y, str(i+1), bbox=dict(facecolor='#ddcba0', alpha=1.0)) 

    ax.plot(xfs,yfs,'g-')
    ax.plot(xhs, yhs, color=tour_color, marker='s', linewidth=3.0) 

    ax.add_patch( mpl.patches.Circle( horseflyinit, radius = 1/60.0,
                                      facecolor= '#D13131', edgecolor='black'   )  )
    fontsize = 20


    ax.set_title( r'Algorithm Used: ' + algo_str +  '\nTour Length: ' \
                   + str(tour_length)[:7], fontdict={'fontsize':fontsize})
    ax.set_xlabel(r'Number of sites: ' + str(len(xsites)) + '\nDrone Speed: ' + str(phi) ,
                      fontdict={'fontsize':fontsize})
    fig.canvas.draw()
    plt.show()
    
    
def draw_phi_mst(ax, phi_mst, inithorseposn, phi):

     # for each tree node draw segments joining to sites (green segs)
     for (nodeidx, nodeinfo) in list(phi_mst.nodes.data()):
         mycoords           = nodeinfo['mycoordinates']
         joined_site_coords = nodeinfo['joined_site_coords'] 

         for site in joined_site_coords:
               ax.plot([mycoords[0],site[0]], [mycoords[1], site[1]], 'g-', linewidth=1.5) 
               ax.add_patch( mpl.patches.Circle( [site[0],site[1]], radius = 0.007, \
                                                 facecolor='blue', edgecolor='black'))

     # draw each tree edge (red segs)
     edges = list(phi_mst.edges.data())
     for (idx1, idx2, edgeinfo) in edges:
          (xn1, yn1) =  phi_mst.nodes[idx1]['mycoordinates']
          (xn2, yn2) =  phi_mst.nodes[idx2]['mycoordinates']
          ax.plot([xn1,xn2],[yn1,yn2], 'ro-' ,linewidth=1.7)

     ax.set_title(r'$\varphi$-MST', fontdict={'fontsize':30})

# Animation routines for classic horsefly

def animateSchedule(schedule_file_name):
     import yaml
     import numpy as np
     import matplotlib.animation as animation
     from matplotlib.patches import Circle
     import matplotlib.pyplot as plt 

     # Set up configurations and parameters for animation and plotting
     
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
     
     # Parse input-output file and set up required data-structures
        
     with open(schedule_file_name, 'r') as stream:
           schedule = yaml.load(stream)

     phi           = float(schedule['phi'])
     inithorseposn = schedule['inithorseposn']

     # Get legs of the horse and fly tours
     horse_tour  = map(np.asarray, schedule['horse_tour']   )
     sites       = map(np.asarray, schedule['visited_sites'])
                

     xhs = [ horse_tour[i][0] for i in range(len(horse_tour))]    
     yhs = [ horse_tour[i][1] for i in range(len(horse_tour))]    
     xfs , yfs = [xhs[0]], [yhs[0]]
     for site, pt in zip (sites,horse_tour[1:]):
              xfs.extend([site[0], pt[0]])
              yfs.extend([site[1], pt[1]])
     fly_tour = map(np.asarray,zip(xfs,yfs))

     horse_legs = zip(horse_tour, horse_tour[1:])
     fly_legs   = zip(fly_tour, fly_tour[1:], fly_tour[2:]) [0::2]

     assert(len(horse_legs) == len(fly_legs))

     # set important meta-data for plot
     ax.set_title("Number of sites: " + str(len(sites)) + "\nTour Length: " +\
                  str(round(utils_algo.length_polygonal_chain(zip(xhs, yhs)),4)), fontsize=20)
     ax.set_xlabel(r"$\varphi$ = " + str(phi), fontsize=20)
     
     # Construct and store every frame of the animation in \verb|ims|
     ims = []
     for horse_leg, fly_leg, leg_idx in zip(horse_legs, \
                                            fly_legs,   \
                                            range(len(horse_legs))):
          debug(Fore.YELLOW + "Animating leg: "+ str(leg_idx) + Style.RESET_ALL)

          # Define function to place points along a leg
             
          def discretize_leg(pts):
             subleg_pts = []
             numpts     = len(pts)

             if numpts == 2:
                 k  = 19 # horse
             elif numpts == 3:
                 k  = 10 # fly

             for p,q in zip(pts, pts[1:]):
                 tmp = []
                 for t in np.linspace(0,1,k): 
                     tmp.append( (1-t)*p + t*q ) 
                 subleg_pts.extend(tmp[:-1])

             subleg_pts.append(pts[-1])
             return subleg_pts
          

          horse_posns = discretize_leg(horse_leg)
          fly_posns   = discretize_leg(fly_leg) 
          assert(len(horse_posns) == len(fly_posns))

          hxs = [xhs[i] for i in range(0,leg_idx+1) ]
          hys = [yhs[i] for i in range(0,leg_idx+1) ]
                
          fxs , fys = [hxs[0]], [hys[0]]
          for site, pt in zip (sites,(zip(hxs,hys))[1:]):
               fxs.extend([site[0], pt[0]])
               fys.extend([site[1], pt[1]])

          number_of_sites_serviced = leg_idx
          for horse_posn, fly_posn, subleg_idx in zip(horse_posns, \
                                                      fly_posns,   \
                                                      range(len(horse_posns))):
               # Render frame and append it to \verb|ims|
                  
               debug(Fore.RED + "Rendering subleg "+ str(subleg_idx) + Style.RESET_ALL)
               hxs1 = hxs + [horse_posn[0]]
               hys1 = hys + [horse_posn[1]]
                              
               fxs1 = fxs + [fly_posn[0]]
               fys1 = fys + [fly_posn[1]]
                                
               # There is a midway update for new site check is site 
               # has been serviced. If so, update fxs and fys
               if subleg_idx == 9:
                   fxs.append(sites[leg_idx][0])
                   fys.append(sites[leg_idx][1])
                   number_of_sites_serviced += 1

               horseline, = ax.plot(hxs1,hys1,'o-', linewidth=5.0, markersize=6, alpha=1.00, color='#d13131')
               flyline,   = ax.plot(fxs1,fys1,'go-', linewidth=1.0, markersize=3)

               objs = [flyline,horseline] 
                
               # Mark serviced and unserviced sites with different colors. 
               # Use https://htmlcolorcodes.com/ for choosing good colors along with their hex-codes.

               for site, j in zip(sites, range(len(sites))):
                   if j < number_of_sites_serviced:       # site has been serviced
                       sitecolor = '#DBC657' # yellowish
                   else:                                  # site has not been serviced
                       sitecolor = 'blue'

                   circle = Circle((site[0], site[1]), 0.015, \
                                   facecolor = sitecolor   , \
                                   edgecolor = 'black'     , \
                                   linewidth=1.4)
                   sitepatch = ax.add_patch(circle)
                   objs.append(sitepatch)

               # Mark initial horse positions with big red circle
               circle = Circle((inithorseposn[0], inithorseposn[1]), 0.020, \
                               facecolor = '#d13131'   , \
                               edgecolor = 'black'     , \
                               linewidth=1.4)
               inithorsepatch = ax.add_patch(circle)
               objs.append(inithorsepatch)

               debug(Fore.CYAN + "Appending to ims "+ Style.RESET_ALL)
               ims.append(objs[::-1])
               
     
     # Write animation of schedule to disk and display in live window
     from colorama import Back 

     debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
     debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)


     #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
     #ani.save(schedule_file_name+'.avi', dpi=150)
     #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

     #plt.show() # For displaying the animation in a live window. 
     

