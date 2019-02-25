
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

xs  = np.linspace(0.20, 0.9, 13)
y1s = np.empty(len(xs)); y1s.fill(0.6)
y2s = np.empty(len(xs)); y2s.fill(0.4)
sites = zip(xs,y1s) + zip(xs,y2s)
inithorseposn = (0.1,0.5)
phi=20.0

utils_algo.print_list(sites)
                                                                                         
collinear_tour  = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi,
                                    write_algo_states_to_disk_p = True,             
                                    animate_schedule_p          = False,             
                                    post_optimizer=chf.algo_exact_given_specific_ordering)  
chf.plotTour(collinear_tour, inithorseposn, phi, 'gincoll')


tsp_ordering_tour = chf.algo_tsp_ordering(sites, inithorseposn, phi,
                                    post_optimizer=chf.algo_exact_given_specific_ordering)  
chf.plotTour(tsp_ordering_tour, inithorseposn, phi, 'tsp')
