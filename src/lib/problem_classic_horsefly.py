    
# Relevant imports for classic horsefly
import sys
import argparse
import pprint as pp
import matplotlib.pyplot as plt
import matplotlib as mpl
from   matplotlib import rc
from colorama import Fore
from colorama import Style
import os
import randomcolor 
import numpy as np
import classic_horsefly as chf
import algo_utils
import graphics_utils

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
                            "  (gl) Greedy (using approximate L1 ordering])  "  +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()

                    # Incase there are patches present from the previous clustering, just clear them
                    clearAxPolygonPatches(ax)

                    if   algo_str == 'e':
                          horseflytour = 
                                 run.getTour( chf.algo_dumb,
                                              phi )
                    elif algo_str == 'k': 
                          horseflytour = 
                                 run.getTour( chf.algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=chf.algo_exact_given_specific_ordering)
                          print " "
                          print Fore.GREEN, answer['tour_points'], Style.RESET_ALL
                    elif algo_str == 'kl':
                          horseflytour = 
                                 run.getTour( chf.algo_kmeans,
                                              phi,
                                              k=2,
                                              post_optimizer=chf.algo_approximate_L1_given_specific_ordering)
                    elif algo_str == 't':
                          horseflytour = 
                                 run.getTour( chf.algo_tsp_ordering,
                                              phi,
                                              post_optimizer=chf.algo_exact_given_specific_ordering)
                    elif algo_str == 'tl':
                          horseflytour = 
                                 run.getTour( chf.algo_tsp_ordering,
                                              phi,
                                              post_optimizer= chf.algo_approximate_L1_given_specific_ordering)
                     elif algo_str == 'g':
                          horseflytour = 
                                 run.getTour( chf.algo_greedy,
                                              phi,
                                              post_optimizer= chf.algo_exact_given_specific_ordering)
                    elif algo_str == 'gl':
                          horseflytour = 
                                 run.getTour( chf.algo_greedy,
                                              phi,
                                              post_optimizer= chf.algo_approximate_L1_given_specific_ordering)
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    #print horseflytour['tour_points']
                    chf.plotTour(ax,horseflytour, run.inithorseposn, phi, algo_str)
                    applyAxCorrection(ax)
                    fig.canvas.draw()
                    
               elif event.key in ['n', 'N', 'u', 'U']: 
                    # Generate a bunch of uniform or non-uniform random points on the canvas
                    numpts = int(sys.argv[1]) 
                    run.clearAllStates()
                    ax.cla()
                                   
                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                    
                    fig.texts = []
                                     
                    import scipy
                    if event.key in ['n', 'N']: # Non-uniform random points
                            run.sites = algo_utils.bunch_of_random_points(numpts)
                    else : # Uniform random points
                            run.sites = scipy.rand(numpts,2).tolist()

                    patchSize  = (xlim[1]-xlim[0])/140.0

                    for site in run.sites:      
                        ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                     facecolor='blue',edgecolor='black' ))

                    ax.set_title('Points : ' + str(len(run.sites)), fontdict={'fontsize':40})
                    fig.canvas.draw()
                    
               elif event.key in ['c', 'C']: 
                    # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
                    
    
    # Set up interactive canvas
    fig, ax =  plt.subplots()
    run = chf.HorseFlyInput()
    #print run
        
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    mouseClick   = wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
    
# Algorithms for classic horsefly
   
   
   
   

