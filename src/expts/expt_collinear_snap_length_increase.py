    
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

def expt(number_of_sites, scheme, inithorseposn, phis, number_of_runs):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  fig, ax = plt.subplots()
  ax.set_title("Tour Length of Collinear Horsefly Tour/Tour Length of Exact Horsefly Tour \n for Greedy Incremental Ordering, $N$="+str(number_of_sites), fontsize=28)
  ax.set_xlabel("Runs", fontsize=25)
  ax.set_ylabel("Tour Length Ratios", fontsize=25)
  #plt.grid(True, linestyle='--')
  plt.tick_params(labelsize=20)
  #ax.set_xticklabels(map(str,range(number_of_runs)))
  #ax.minorticks_on()

  ax.set_ylim([0.75,1.75])
  
  for phi in phis:
     ratios = []
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

         ratios.append(collinear_tour['tour_length_with_waiting_time_included']/
                    collinear_tour_after_slsqp['tour_length_with_waiting_time_included'])
      
         print "..........................................................................................................."
         print "Collinear Tour Length            : ", collinear_tour['tour_length_with_waiting_time_included']
         print "Collinear Tour Length After SLSQP: ", collinear_tour_after_slsqp['tour_length_with_waiting_time_included']

     plt.plot(range(number_of_runs), ratios, "o-", label=r"$\varphi$="+str(phi))
     
  ax.legend(prop={'size': 20})
  plt.show()


if __name__ == "__main__":
    expt(number_of_sites=10, scheme='uniform', inithorseposn=(0.5,0.5), \
         phis=[3.0, 6.0, 12.0, 24.0], number_of_runs=40)

