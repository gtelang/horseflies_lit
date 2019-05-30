
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import Fore, Back, Style
import argparse 
sys.path.append('../lib')
import problem_one_horse_multiple_flies as ohmf
import utils_graphics
import utils_algo
import scipy
import matplotlib as mpl

def expt(number_of_sites, scheme, inithorseposn, phi):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  fig, ax = plt.subplots()
  ax.set_title("Tour Length v/s Number of Drones for Earliest Capture Heuristic, $N$="+str(number_of_sites) + " ("+scheme+")", fontsize=28)
  ax.set_xlabel("Number of Drones", fontsize=25)
  ax.set_ylabel("Tour Length", fontsize=25)
  plt.grid(True, linestyle='--')
  
  textstr = r"$\varphi$=" + str(phi)
  ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=40,
        verticalalignment='top')
  plt.tick_params(labelsize=20)

  for i in range(5):
      # Generate a point set of size $N$
      if scheme == 'nonuniform': 
         sites = utils_algo.bunch_of_non_uniform_random_points(number_of_sites)
      elif scheme == 'uniform' : 
         sites = scipy.rand(number_of_sites,2).tolist()
      else:
          print "scheme not recognized"
          sys.exit()

      ks           = range(2,40,4)
      tour_lengths = []
      for k in ks:
          tour = ohmf.algo_greedy_earliest_capture(sites=sites, 
                                                   inithorseposn=inithorseposn, 
                                                   phi=phi, 
                                                   number_of_flies=k,
                                                   write_algo_states_to_disk_p = False,\
                                                   write_io_p                  = False,\
                                                   animate_tour_p              = False)
          horse_traj_pts = map(lambda x: x[0], tour['horse_trajectory'])
          tour_lengths.append(utils_algo.length_polygonal_chain(horse_traj_pts))

      ax.plot(ks, tour_lengths, "o-", linewidth=4, markersize=7, label="Cloud "+str(i))
  
  ax.legend(prop={'size': 20})
  plt.show()


if __name__ == "__main__":
    expt(number_of_sites=100, scheme='uniform', inithorseposn=(0.5,0.5), phi=3.0)
