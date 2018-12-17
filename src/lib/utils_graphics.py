

from matplotlib import rc
from colorama import Fore
from colorama import Style
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time

xlim, ylim = [0,1], [0,1]

def applyAxCorrection(ax):
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
      ax.set_aspect(1.0)

def clearPatches(ax):
    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()

    # Remove line patches. These get inserted during the r=2 case,
    # For some strange reason matplotlib does not consider line objects
    # as patches.
    ax.lines[:]=[]

    #pp.pprint (ax.patches) # To verify that none of the patches are
    # polyon patches corresponding to clusters.
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()

    # Remove line patches. These get inserted during the r=2 case,
    # For some strange reason matplotlib does not consider line objects
    # as patches.
    ax.lines[:]=[]

    # To verify that none of the patches 
    # are polyon patches corresponding 
    # to clusters.
    #pp.pprint (ax.patches) 
    applyAxCorrection(ax)

## Also modify to enter initial position of horse and fly
def wrapperEnterRunPoints(fig, ax, run):
    """ Create a closure for the mouseClick event.
    """
    def _enterPoints(event):
        if event.name     == 'button_press_event'      and \
           (event.button   == 1 or event.button == 3)  and \
           event.dblclick == True                      and \
           event.xdata    != None                      and \
           event.ydata    != None:

             if event.button == 1:        
               newPoint = (event.xdata, event.ydata)
               run.sites.append( newPoint  )
               patchSize  = (xlim[1]-xlim[0])/140.0
   
               ax.add_patch( mpl.patches.Circle( newPoint,
                                              radius = patchSize,
                                              facecolor='blue',
                                              edgecolor='black'   )  )
               ax.set_title('Points Inserted: ' + str(len(run.sites)), \
                             fontdict={'fontsize':40})


             if event.button == 3:        
                 inithorseposn = (event.xdata, event.ydata)
                 run.inithorseposn = inithorseposn  
                 patchSize  = (xlim[1]-xlim[0])/70.0

                 # TODO: remove the previous red patches, 
                 # which containg ht eold position
                 # of the horse and fly. Doing this is 
                 # slightly painful, hence keeping it
                 # for later
                 ax.add_patch( mpl.patches.Circle( inithorseposn,
                                                   radius = patchSize,
                                                   facecolor= '#D13131', #'red',
                                                   edgecolor='black'   )  )
                 
             # It is inefficient to clear the polygon patches inside the
             # enterpoints loop as done here.
             # I have just done this for simplicity: the intended behaviour
             # at any rate, is
             # to clear all the polygon patches from the axes object,
             # once the user starts entering in MORE POINTS TO THE CLOUD
             # for which the clustering was just computed and rendered.
             # The moment the user starts entering new points,
             # the previous polygon patches are garbage collected. 
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPoints
