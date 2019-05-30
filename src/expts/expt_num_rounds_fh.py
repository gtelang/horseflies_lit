                                                                                                 
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
                              
from matplotlib.ticker import FormatStrFormatter

# compare makespans for greedy_nn_concentric_routing 
# (gncr) vs greedy_kinetic_tsp (gkin) we do this first 
# for many randomly generated point clouds of size 100
num_reps          = 50
num_pts_per_cloud = [10,20,40, 80, 160]
numrounds_list    = [[] for n in range(len(num_pts_per_cloud))]

inithorseposn =  np.asarray([0.5, 0.5])
phi           =  8.0

m = np.inf
M = -np.inf

for i in range(len(num_pts_per_cloud)):
    for j in range(num_reps):
        print i, j
        sites = np.random.rand(num_pts_per_cloud[i],2)
        htraj_fh, fly_trajs_fh, numrounds  = rhf.algo_proceed_to_farthest_drone(sites, inithorseposn, phi, \
                                                                                write_algo_states_to_disk_p = False, write_io_p = False, 
                                                                                animate_tour_p              = False, plot_tour_p= False) 
        numrounds_list[i].append(numrounds)

        if numrounds < m:
            m = numrounds
        if numrounds > M:
            M = numrounds


ymin = 1
ymax = 6
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots()
ax.set_yticks(range(ymin,ymax+1), minor=False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel("Runs", fontsize=25)
ax.set_ylabel("Number of Rounds", fontsize=25)
ax.set_ylim(ymin,ymax)
plt.grid(True, linestyle='--')
plt.tick_params(labelsize=20)

for i in range(len(numrounds_list)):
    plt.plot(range(num_reps), numrounds_list[i], "o-", markersize=7, linewidth=4, label=str(num_pts_per_cloud[i]))

ax.set_title(" Total number of rounds to complete the fh heuristic "  , fontsize=30)
plt.figtext(0.5, 0.01, "Number of drones per run: " + str(num_pts_per_cloud) + "\hspace{1cm}" + r"$\varphi$: " + str(phi), horizontalalignment='center', fontsize=23)
ax.legend(prop={'size': 20})
plt.show()
