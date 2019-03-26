                                                                                                 
import sys, os                                                                                   
import numpy as np                                                                               
import matplotlib.pyplot as plt                                                                  
from matplotlib import rc                                                                        
from colorama import Fore, Back, Style                                                           
import argparse                                                                                  
sys.path.append('../lib')                                                                        
import problem_reverse_horsefly as rhf                                                           
import utils_graphics                                                                            
import utils_algo                                                                                
import scipy                                                                                     
import matplotlib as mpl                                                                         
                              
# compare makespans for greedy_nn_concentric_routing 
# (gncr) vs greedy_kinetic_tsp (gkin) we do this first 
# for many randomly generated point clouds of size 100
num_reps   = 100
num_pts_per_cloud = 60

mspans_gncr = []
mspans_gkin = []

inithorseposn = np.asarray([0.0, 0.0])
phi = 0.5

for i in range(num_reps):
    sites = np.random.rand(num_pts_per_cloud,2)

    htraj_gncr, fly_trajs_gncr = rhf.algo_greedy_nn_concentric_routing(sites, inithorseposn, phi, \
                                      shortcut_squiggles_p        = True, write_algo_states_to_disk_p = False,\
                                      write_io_p                  = False,animate_tour_p              = False,\
                                      plot_tour_p                 = False) 

    htraj_gkin, fly_trajs_gkin = rhf.algo_greedy_concentric_kinetic_tsp(sites, inithorseposn, phi, \
                                      write_algo_states_to_disk_p = False, write_io_p = False, 
                                      animate_tour_p              = False, plot_tour_p= False) 
   
    mspan_gncr, _ = rhf.makespan([htraj_gncr])
    mspan_gkin, _ = rhf.makespan([htraj_gkin])

    mspans_gncr.append(mspan_gncr)
    mspans_gkin.append(mspan_gkin)
    

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots()
ax.set_xlabel("Runs", fontsize=25)
ax.set_ylabel("Makespan of gncr and gkin", fontsize=25)
plt.grid(True, linestyle='--')
plt.tick_params(labelsize=20)

plt.plot(range(num_reps), mspans_gncr, 'go-', label=r"gncr")
plt.plot(range(num_reps), mspans_gkin, 'ro-', label=r"gkin")

ax.set_title("Number of drones per run: " + str(num_pts_per_cloud), fontsize=15)

ax.legend(prop={'size': 20})
plt.show()
