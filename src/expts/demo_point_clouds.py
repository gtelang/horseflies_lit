# Make plots for this for the demo. Use this for about 100 sites
# so that the algorithms are scalable and you can obtain enough
# data for the purpose of plots. 
                                                                                                 
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
import data_sets as ds                      
        
num_pts_per_cloud = 100
cloud_type        = 'grid' # uni, annulus, ball, clusunif, normal, spokes, grid
inithorseposn     = np.asarray([0.5, 0.5])
phi               = 3.0
sites             = ds.genpointset(num_pts_per_cloud,cloud_type)
assert len(sites) == num_pts_per_cloud,\
      "Number of sites should be equal to num_pts_per_cloud "


ds.plotpoints(sites)



# hftour = chf.algo_greedy(sites, inithorseposn, phi, 
#                          write_algo_states_to_disk_p = False   ,
#                          animate_schedule_p          = False   , 
#                          post_optimizer              = chf.algo_exact_given_specific_ordering)


# chf.plotTour(hftour, inithorseposn, phi, 'greedy', tour_color='#d13131')
# plt.show()
