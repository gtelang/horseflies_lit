

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
    ax.lines[:]=[]
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)
def wrapperEnterRunPoints(fig, ax, run):
    def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert blue circle representing a site
                   
                 newPoint = (event.xdata, event.ydata)
                 run.sites.append( newPoint  )
                 patchSize  = (xlim[1]-xlim[0])/140.0
                    
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='blue', edgecolor='black'  ))
                 ax.set_title('Points Inserted: ' + str(len(run.sites)), \
                              fontdict={'fontsize':40})
                 

             elif event.button == 3:  
                 # Insert big red circle representing initial position of horse and fly
                  
                 inithorseposn     = (event.xdata, event.ydata)
                 run.inithorseposn = inithorseposn  
                 patchSize         = (xlim[1]-xlim[0])/70.0

                 ax.add_patch( mpl.patches.Circle( inithorseposn,radius = patchSize,
                                                   facecolor= '#D13131', edgecolor='black' ))
                 

             # Clear polygon patches and set up last minute \verb|ax| tweaks
                
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()
             

    return _enterPoints
def get_random_color(sat=0.7,val=0.7):
    def hsv_to_rgb(h, s, v):
          h_i = int((h*6))
          f   = h*6 - h_i
          p   = v * (1 - s)
          q   = v * (1 - f*s)
          t   = v * (1 - (1 - f) * s)
  
          if h_i==0:
               r, g, b = v, t, p 
          elif h_i==1:  
               r, g, b = q, v, p 
          elif h_i==2:
               r, g, b = p, v, t 
          elif h_i==3:
               r, g, b = p, q, v 
          elif h_i==4: 
               r, g, b = t, p, v 
          elif h_i==5:
               r, g, b = v, p, q 

          return [int(r*256), int(g*256), int(b*256)]
    return hsv_to_rgb(np.random.rand(), sat, val)
