import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':15})
rc('text', usetex=True)

import matplotlib as mpl
import sys, os, time
import itertools
from colorama import Fore, Style
import utils_graphics
import randomcolor

rand_color = randomcolor.RandomColor()

# Local data-structures

 
class SegmentHorsefly:
      def __init__(self, phi, sites=[], inithorseposn=[]):
           self.sites                = sites
           self.inithorseposn        = inithorseposn
           self.phi                  = phi 


      def clearAllStates (self):
          self.sites           = []
          self.inithorseposn   = None
          self.phi             = float(raw_input("Enter drone speed: "))

      def getTour(self, fig, ax):
          pass

def applyAxCorrection(ax):
      ax[0].set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
      ax[0].set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
      ax[0].set_aspect(1.0)

def clearPatches(ax):
    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax[0].patches)), ax[0].patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax[0].lines[:]=[]
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax[0].patches)), ax[0].patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax[0].lines[:]=[]
    applyAxCorrection(ax)


# Implementation of the key-press handler

def wrapperkeyPressHandler(fig,ax, run): 
      def _keyPressHandler(event):
                      
             if event.key in ['c', 'C']: 
                   # Clear canvas and states of all objects

                    ax[0].cla()
                    ax[1].cla()
                    run.clearAllStates()


                    applyAxCorrection(ax)
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
             

             elif event.key in ['a','A']: # run the greedy algorithm  to get a tour
                   run.getTour(fig,ax)


      return _keyPressHandler

# Implementation of \verb|wrapperEnterPoints|
   
def wrapperEnterPoints(fig,ax,run):
    def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert blue circle representing a site
                 newPoint = (event.xdata, event.ydata)
                 run.sites.append( newPoint  )
                 patchSize  = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/40.0

                 sitecolor = rand_color.generate()[0]

                 ax[0].add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor=sitecolor, edgecolor='black'  ))
                 ax[0].set_title(r'Number of sites : ' + str(len(run.sites)), \
                              fontdict={'fontsize':20})

                 ax[1].set_title(r'Drone speed $\varphi$: ' + str(run.phi))

                 x,y = newPoint
                 asp  = np.linspace(0,2,100) # beginning positions of truck
                 lasp = []
                 for a in asp:
                     S   = np.sqrt((x-a)**2+y**2)
                     S_1 = phi*a + S
                     
                     l = ((phi*S_1-x) + np.sqrt ((phi*S_1-x)**2 + (phi**2-1)*(x**2+y**2 -S_1**2)))
                     l /= (phi**2 - 1)
                     l -= a 
                     lasp.append(l)
                 lasp = np.asarray(lasp)

                 ax[1].plot(asp,lasp,'-', color=sitecolor)
                 ax[1].set_xlabel(r"$a$")
                 ax[1].set_ylabel(r"$l(a)$")
                 ax[1].set_xlim(0,1)
                 #ax[1].set_ylim(0,1)
                 fig.canvas.draw()
                
    return _enterPoints


# if __name__ == "__main__":
#     # Body of main function
    
#     fig, ax =  plt.subplots(1,2)
#     phi=float(raw_input("Enter drone speed: "))
#     run = SegmentHorsefly(phi=phi)
            
#     ax[0].set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
#     ax[0].set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
#     ax[0].set_aspect(1.0)

#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
              
#     mouseClick = wrapperEnterPoints (fig,ax, run)
#     fig.canvas.mpl_connect('button_press_event' , mouseClick )
             
#     keyPress   = wrapperkeyPressHandler(fig,ax, run)
#     fig.canvas.mpl_connect('key_press_event', keyPress )

#     plt.show()
    
if __name__ == "__main__":
      # Generate the data
      x,y = 10.0, 10.0             # position of site
      phi = 3.0                  # speed ratio

      asp  = np.linspace(0,20,100) # beginning positions of truck

      # Calculate the meeting points of the drone with the truck
      lasp = []
      for a in asp:
            S   = np.sqrt((x-a)**2+y**2)
            S_1 = phi*a + S
    
            l = ((phi*S_1-x) + np.sqrt ((phi*S_1-x)**2 + (phi**2-1)*(x**2+y**2 -S_1**2)))
            l /= (phi**2 - 1)
            l -= a 
            lasp.append(l)
      lasp = np.asarray(lasp)

      fig, ax = plt.subplots() # plot the data

      ax.plot(asp,lasp,'r-')
      ax.set_xlabel(r"$a$")
      ax.set_ylabel(r"$l(a)$")

      plt.xticks(np.arange(min(asp) , max(asp)+1 , 2.0))
      plt.yticks(np.arange(min(lasp), max(lasp)+1, 2.0))
      plt.grid(b=True,linestyle='--')
      plt.show()
