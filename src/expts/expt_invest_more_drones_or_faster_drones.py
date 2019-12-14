import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import Fore, Back, Style
import argparse 
sys.path.append('../lib')
import problem_one_horse_multiple_flies as ohmf
import problem_classic_horsefly as chf
import utils_graphics
import utils_algo
import scipy
import matplotlib as mpl

def expt(number_of_sites, inithorseposn, phi):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')

  fig, ax = plt.subplots()
  ax.set_xlabel("Number of Drones of Speed " + r"$\varphi$" + "(for blue curve) \nSpeed of Single Fast Drone (for black curve)", fontsize=25)
  ax.set_ylabel("Tour Length of Truck", fontsize=25)
  ax.set_title("Drop in tour length of truck for Greedy\n" + r"Number of sites $N$=" + str(number_of_sites) + " (uniform)" , fontsize=30)
  plt.grid(True, linestyle='--')

  textstr = r"$\varphi$=" + str(phi)
  ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=35,verticalalignment='center')
  plt.tick_params(labelsize=20)

  sites              = scipy.rand(number_of_sites,2).tolist()
  ks                 = range(2,40,4)
  tour_single_greedy = []
  tour_ohmf_lengths  = []


  for k in ks:
          
          tour_single_drone =  chf.algo_greedy(sites, inithorseposn, phi=k, 
                                               write_algo_states_to_disk_p = False   ,
                                               animate_schedule_p          = False   , 
                                               post_optimizer              = chf.algo_exact_given_specific_ordering)


          tour_ohmf = ohmf.algo_earliest_capture_postopt(sites=sites, 
                                                   inithorseposn=inithorseposn, 
                                                   phi=phi, 
                                                   number_of_flies=k,
                                                   write_algo_states_to_disk_p = False,\
                                                   write_io_p                  = False,\
                                                   animate_tour_p              = False)
          
          # single drone truck tour length
          tour_single_greedy.append(tour_single_drone['tour_length_with_waiting_time_included'])

          # multuple drones truck tour length
          horse_traj_ohmf_pts = map(lambda x: x[0], tour_ohmf['horse_trajectory'])
          tour_ohmf_lengths.append(utils_algo.length_polygonal_chain(horse_traj_ohmf_pts))

  #................................................... 
  ax.plot(ks, tour_single_greedy, "ko--", linewidth=2.5)
  ax.plot(ks, tour_ohmf_lengths, "o-", linewidth=2.5)
  plt.show()
  #................................................... 

if __name__ == "__main__":
    expt(number_of_sites=100, inithorseposn=(0.5,0.5), phi=12.0)
