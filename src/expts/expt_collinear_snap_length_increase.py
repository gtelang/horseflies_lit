import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import Fore, Back, Style
import argparse 
sys.path.append('../lib')
import problem_classic_horsefly as chf
import utils_graphics
import utils_algo
import scipy
import matplotlib as mpl

def expt(number_of_sites, scheme, inithorseposn, phi, number_of_runs):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  fig, ax = plt.subplots()
  ax.set_title("Collinear Tour Length / Exact Tour Length (Greedy Incremental Ordering), $N$="+str(number_of_sites) + " ("+scheme+")", fontsize=28)
  ax.set_xlabel("Tour Length Ratios", fontsize=25)
  ax.set_ylabel("Run Number", fontsize=25)
  plt.grid(True, linestyle='--')
  plt.tick_params(labelsize=20)
  
  
  for i in range(number_of_runs):

      # Generate a point set of size $N$                                                         
      if scheme == 'nonuniform':                                                                 
         sites = utils_algo.bunch_of_non_uniform_random_points(number_of_sites)                  
      elif scheme == 'uniform' :                                                                 
         sites = scipy.rand(number_of_sites,2).tolist()                                          
      else:                                                                                      
          print "scheme not recognized"                                                          
          sys.exit()                              
      
      collinear_tour             = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi,\
                                                write_algo_states_to_disk_p = False,
                                                animate_schedule_p          = False,
                                                post_optimizer=None)
      
      collinear_tour_after_slsqp = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi,\
                                                write_algo_states_to_disk_p = False,
                                                animate_schedule_p          = False,
                                                post_optimizer=chf.algo_exact_given_specific_ordering)

      print "Collinear Tour Length            : ", collinear_tour['tour_length_with_waiting_time_included']
      print "Collinear Tour Length After SLSQP: ", collinear_tour_after_slsqp['tour_length_with_waiting_time_included']

  plt.show()


if __name__ == "__main__":
    expt(number_of_sites=10, scheme='uniform', inithorseposn=(0.5,0.5), \
         phi=3.0, number_of_runs=3)
