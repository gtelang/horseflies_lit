    
# Relevant imports for classic horsefly

from matplotlib import rc
from colorama import Fore
from colorama import Style
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import inspect 
import itertools
import logging
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

# Set up logging information relevant to this module
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
                            "  (gl)  Greedy (using approximate L1 ordering])\n"  +\
                            "  (ginc) Greedy Incremental  "  +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()

                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

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
                                              
                    elif algo_str == 'ginc':
                          horseflytour = \
                                 run.getTour( algo_greedy_incremental_insertion,
                                              phi )

                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    #print horseflytour['tour_points']
                    plotTour(ax,horseflytour, run.inithorseposn, phi, algo_str)
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
                    if event.key in ['n', 'N']: # Non-uniform random points
                            run.sites = utils_algo.bunch_of_random_points(numpts)
                    else : # Uniform random points
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
   
def algo_dumb(sites, horseflyinit, phi):
    """ For each of the n factorial ordering of sites
    find the ordering which gives the smallest horsefly 
    tour length
    """
    
    tour_length_fn = tour_length(horseflyinit)

    best_tour = algo_exact_given_specific_ordering(sites, horseflyinit, phi)

    i = 0
    for sites_perm in list(itertools.permutations(sites)):
        print "Testing a new permutation ", i, " of the sites"; i = i + 1
          
        #tour_for_current_perm = algo_exact_given_specific_ordering (sites_perm, \
        #                                                             horseflyinit, phi) 
        tour_for_current_perm = algo_exact_given_specific_ordering (sites_perm, \
                                                                    horseflyinit, phi) 
        if tour_length_fn(utils_algo.flatten_list_of_lists(tour_for_current_perm ['tour_points']) ) \
         < tour_length_fn(utils_algo.flatten_list_of_lists(            best_tour ['tour_points']) ):
                best_tour = tour_for_current_perm
                
                print Fore.RED + "Found better tour!" + Style.RESET_ALL

    #print Fore.RED + "\nHorse Waiting times are ",\ 
    #       best_tour['horse_waiting_times'] , \
    #       Style.RESET_ALL
    return best_tour

   
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
                  # Warning this is inefficient!!! I am rebuilding the 
                  # kd-tree at each step. Right now, I am only doing 
                  # this for convenience.
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

# ALGORITHMS FOR SINGLE HORSE SINGLE FLY SERVICING THE SITES IN THE GIVEN ORDER
def algo_exact_given_specific_ordering (sites, horseflyinit, phi):
    """ Use the *given* ordering of sites to compute a good tour 
    for the horse.
    """
    def ith_leg_constraint(i, horseflyinit, phi, sites):
        """ For the ith segment of the horsefly tour
        this function returns a constraint function which 
        models the fact that the time taken by the fly 
        is equal to the time taken by the horse along 
        that particular segment.
        """
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
        """ Given input data, of the problem generate the 
        constraint list for each leg of the tour. The number
        of legs is equal to the number of sites for the case 
        of single horse, single drone
        """
        cons = []
        for i in range(len(sites)):
            cons.append( { 'type':'eq',
                            'fun': ith_leg_constraint(i,horseflyinit,phi, sites) } )
        return cons


    
    cons = generate_constraints(horseflyinit, phi, sites)
    # Since the horsely tour lies inside the square,
    # the bounds for each coordinate is 0 and 1
    #x0 = np.empty(2*len(sites))
    #x0.fill(0.5) # choice of filling vector with 0.5 is arbitrary

    x0 = utils_algo.flatten_list_of_lists(sites) # the initial choice is just the sites
    assert(len(x0) == 2*len(sites))
    x0 = np.array(x0)
    
    sol = minimize(tour_length(horseflyinit), x0, method= 'SLSQP', \
                               constraints=cons, options={'maxiter':500})
    
    tour_points = utils_algo.pointify_vector(sol.x)

    # return the waiting times for the horse
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
       
       best_table_entry = min(delta_increase_least_table, key = lambda x: x['delta_increase'])
                
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
                                      write_algo_states_to_disk_p = True   ,
                                      post_optimizer              = None):
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

      algo_state_counter = 0 
      
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
         
         # Write algorithm's current state to file
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
      horse_waiting_times = np.zeros(len(sites))
      return {'tour_points'                : insertion_policy.horse_tour[1:],
              'horse_waiting_times'        : horse_waiting_times, 
              'site_ordering'              : insertion_policy.visited_sites,
              'tour_length_with_waiting_time_included': \
                                             tour_length_with_waiting_time_included(\
                                                          insertion_policy.horse_tour[1:], \
                                                          horse_waiting_times, \
                                                          inithorseposn)}
      

# Plotting routines for classic horsefly
def plotTour(ax,horseflytour, horseflyinit, phi, algo_str, tour_color='#d13131'):
   
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


    fontsize = 20
    tnrfont = {'fontname':'Times New Roman'}
    ax.set_title( r'Algorithm Used: ' + algo_str +  '\nTour Length: ' \
                    + str(tour_length)[:7], fontdict={'fontsize':fontsize}, **tnrfont)
    ax.set_xlabel(r'Number of sites: ' + str(len(xsites)) + '\nDrone Speed: ' + str(phi) ,
                  fontdict={'fontsize':fontsize}, **tnrfont)

