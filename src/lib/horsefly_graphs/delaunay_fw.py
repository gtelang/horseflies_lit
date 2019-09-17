from CGAL.CGAL_Kernel import Point_2, Segment_2, Iso_rectangle_2
from CGAL.CGAL_Kernel import do_intersect, intersection
from colorama import Fore, Style
from colorama import Fore, Style
from matplotlib import rc
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Delaunay
import geometric_median as gm
import itertools
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
import sys, os, time
import triangle_incircle as ti
import utils_graphics, utils_algo

def cleanup(pts,tol=1e-4):
    
  counter = 0  
  while counter <= 5:
    pts    = np.asarray(pts)
    numpts = len(pts)
    for i in range(numpts-1):
        dmin = np.inf
        jmin = i+1

        for j in range(i+1,numpts):
            d = np.linalg.norm(pts[i]-pts[j])
            if d < dmin:
                jmin = j
 
        z         = pts[i+1]
        pts[i+1]  = pts[jmin]
        pts[jmin] = z
        
    simflags = [0]    
    for i in range(1,numpts):
        if np.linalg.norm(pts[i]-pts[i-1]) < tol:
            simflags.append(1)

    clean_pts = [  pt for (pt,sf) in zip(pts,simflags) if sf == 0    ]
    
    counter += 1

  return np.asarray(clean_pts)  


def sum_dists_to_pt(pts, p):
  pts = np.asarray(pts)
  p   = np.asarray(p)
  return  sum([np.linalg.norm(p-pt) for pt in pts])

def center_of_mass(pts):
  pts = np.asarray(pts)
  #print "-------------"
  #print pts
  #print sum(pts[:,0]), sum(pts[:,1])
  #print "-------------"
  return np.asarray( [sum(pts[:,0])/len(pts), sum(pts[:,1])/len(pts)]  )



 

# Local data-structures
class HorseflyInputGraph:
      def __init__(self, sites=[], inithorseposn=[]):
           self.sites                = sites

      def clearAllStates (self):
          self.sites           = []
          
      def maxDispDelTri(self,fig,ax):
          
          def get_max_disp_vc_nodes(graph, points):
                """ Get the maximally dispersed vertex cover for the supplied graph of the delaunay triangulation
                on the graph. Returns a list of indices. Distances are measured in Euclidean Length between points
                """
                
                def exists_disp_vertex_cover_more_than_bin_guess(graph, points,d):
                     """ Check if there exists a vertex cover in the graph whose
                     dipsersion is at least d. We solve this using 2-SAT as 
                     described in the paper by Arkin and Hassin Theorem 4.1, Page 10, 
                     \"Minimum Diameter Covering Problems\" that can be downloaded
                     from http://www.ams.sunysb.edu/~estie/papers/diam.ps.gz
                     """
                     from satispy import Variable, Cnf
                     from satispy.solver import Minisat

                     numpts = len(points)
                     exp    = Cnf()

                     #------------------------------------------------
                     # Set up variables x_i for each node of the graph
                     #------------------------------------------------
                     xs = []
                     for idx in range(numpts):
                          xs.append(Variable( 'x_'+str(idx) ))
                     print "     Created Variables...."
                     #-------------------------------
                     # Set up clauses for each graph
                     #-------------------------------
                     for (i,j) in  list(graph.edges):
                             c = Cnf()
                             c = xs[i] | xs[j]
                             exp &= c

                     print "     Created Edge Clauses..."
                     #--------------------------------------------------------------------
                     # For any pair of vertices k and l whose distance from one another 
                     # is smaller than d we know we can choose at most one of the vertices
                     #--------------------------------------------------------------------
                     for i in range(numpts):
                         for j in range(i+1, numpts):
                             if np.linalg.norm(points[i]-points[j]) < d:
                                    c = Cnf()
                                    c = -xs[i] | -xs[j]
                                    exp &= c
                     print "     Created All Pair Clauses....."
                     #--------------------------------
                     # Solve the system with Minisat
                     #--------------------------------
                     solver = Minisat()
                     solution = solver.solve(exp)

                     if solution.success:

                            print      Fore.CYAN, "    2-SAT Solution found!", Style.RESET_ALL
                            return (True, [idx for idx in range(numpts) if solution[xs[idx]] == True])
                     else: 
                            print  Fore.RED, "    2-SAT Solution cannot be found", Style.RESET_ALL
                            return (False, [])




                tol               = 1e-6      # configurable
                bin_low, bin_high = 0.0 , 2.0 # initial interval limits for binary search
                old_disp_idxs     = range(len(points)) 

                while abs(bin_high - bin_low) > tol:
                    #-----------------------------
                    # Common sense sanity checks
                    #-----------------------------
                    assert bin_high >= bin_low 

                    bin_guess = (bin_high + bin_low)/2.0
                    
                    print "\n" + Fore.GREEN, "Binary search Interval Limits are ", "[", bin_low, " , ", bin_high, "]"
                    print Fore.GREEN, "Guessed dispersion value currently is ", bin_guess, Style.RESET_ALL
                    print Fore.GREEN, "Checking for vertex cover with ", bin_guess, " dispersion....."
                    #--------------------------------------------------
                    # Check if there exists a dispersed vertex cover 
                    # that is <= or > bin_guess and adjust interval limits 
                    # of binary search accordingly.
                    #--------------------------------------------------
                    [exist_flag_p, disp_idxs] = exists_disp_vertex_cover_more_than_bin_guess(graph,points,bin_guess)

                    if exist_flag_p:
                          bin_low = bin_guess
                          old_disp_idxs = disp_idxs
                    else:
                          bin_high = bin_guess 

                #---------------------------------------------------------          
                # Return the indices of the points corresponding to the 
                # dispersed cover along with the dispersion value.
                #---------------------------------------------------------
                return (old_disp_idxs, bin_guess)


          os.system("rm max_disp_delaunay_tris/*")  
          points          = np.asarray(self.sites)
          original_points = points
          original_numpts = len(points)


          filenum = 0
          plt.cla()
          plt.grid(linestyle='--')
          ax.set_aspect(aspect=1.0)
          ax.set_xlim([0.0,1.0])
          ax.set_ylim([0.0,1.0])

          ax.set_title("Number of red points: " + str(len(points)))
          ax.plot(points[:,0], points[:,1], 'o', markerfacecolor='r', markersize=10)
          plt.savefig('max_disp_delaunay_tris/myplot_' + str(filenum).zfill(4) + '.png', bbox_inches='tight', dpi=200)
          

          while len(points) >= 10 :
              print Fore.YELLOW
              print "==============================================================================="
              print "ITERATION ", filenum
              print "===============================================================================", Style.RESET_ALL
              #---------------------------------------------
              # Increment file counter for next iteration
              #---------------------------------------------
              filenum += 1

              #-----------------------------------------------------------------------
              # Clear plot and set up basic things needed for plotting into a new file
              #-----------------------------------------------------------------------
              plt.cla()
              plt.grid(linestyle='--')
              ax.set_aspect(aspect=1.0)
              ax.set_xlim([0.0,1.0])
              ax.set_ylim([0.0,1.0])
              ax.scatter(original_points[:,0], original_points[:,1],alpha=0.2)

              #---------------------------------------
              # Construct the delaunay triangulation
              #---------------------------------------
              tri     = Delaunay(points)
              numtris = len(tri.simplices)
              #ax.triplot(points[:,0], points[:,1], tri.simplices.copy())

              del_tri_edge_list = []
              for simplex, fidx in zip(tri.simplices,range(numtris)):
                  [i,j,k] = simplex
                  
                  del_tri_edge_list.append([[min(i,j), max(i,j)], fidx])
                  del_tri_edge_list.append([[min(j,k), max(j,k)], fidx])
                  del_tri_edge_list.append([[min(k,i), max(k,i)], fidx])

              del_tri_edge_list.sort()
              from itertools import groupby
              del_tri_edge_list = [elt[0] for elt in groupby(del_tri_edge_list)]

              #-----------------------------------------------------------
              # Create a networx graph corresponding to the triangulation
              #-----------------------------------------------------------
              delgraph = nx.Graph()
              delgraph.add_edges_from([(i,j) for [[i,j],_] in del_tri_edge_list])

              #----------------------------------------------------------
              # get indices of the dispersed nodes in the graph
              #----------------------------------------------------------
              max_disp_vc_nodes, dispersion = get_max_disp_vc_nodes(delgraph, points) 
              points                        = np.asarray([points[idx] for idx in max_disp_vc_nodes])

              tri     = Delaunay(points)
              numtris = len(tri.simplices)
              
              #ax.triplot(points[:,0], points[:,1], tri.simplices.copy())

              #------------------------------------------------------------------
              # Render the dispersed nodes in a different color and save to disk
              #------------------------------------------------------------------
              ax.plot(points[:,0], points[:,1], 'o', markerfacecolor='r', markersize=10)
              ax.set_title("Number of red points: " + str(len(points)))
              plt.savefig('max_disp_delaunay_tris/myplot_' + str(filenum).zfill(4) + '.png', bbox_inches='tight', dpi=200)


      def runRecDel(self,fig, ax):

          os.system("rm fermat_weber_plots/*")  
          points = np.asarray(self.sites)
          
          original_points = points
          original_numpts = len(points)
          glob_fw_center = np.asarray(gm.geometric_median(points))
          glob_com = center_of_mass(original_points)
          
          filenum = 0
          while len(points) >= 6 and len(points) <= 10000000:

              plt.cla()
              plt.grid(linestyle='--')
              ax.set_aspect(aspect=1.0)
              ax.set_xlim([0.0,1.0])
              ax.set_ylim([0.0,1.0])

              print "Number of points is now: ", len(points)

              # Construct and plot the delauny triangulation
              tri = Delaunay(points)

              ax.triplot(points[:,0], points[:,1], tri.simplices.copy())
              ax.plot(points[:,0], points[:,1], 'o')

              # Get the fermat-weber center of each face of the triangulation
              fw_centers = []
              for tri in points[tri.simplices]:
                  [p,q,r] = tri
                  #fwc = gm.geometric_median([p,q,r])
                  #r, fwc = ti.triangle_incircle(np.asarray([[p[0],q[0],r[0]], [p[1],q[1],r[1]]]))
                  fwc = np.asarray([ (p[0]+q[0]+r[0])/3.0, (p[1]+q[1]+r[1])/3.0])
                  #print Fore.YELLOW, "fwc=", fwc, Style.RESET_ALL
                  fw_centers.append(fwc)

              if len(fw_centers) > original_numpts    : 
                   fw_centers_idxs = np.random.choice(range(len(fw_centers)), original_numpts, replace=False)  
                   fw_centers = [fw_centers[idx] for idx in fw_centers_idxs]

              fw_centers = np.asarray(fw_centers)

              #print Fore.RED, "fw_centers", fw_centers, Style.RESET_ALL
              

              # Plot these centers
              ax.plot(fw_centers[:,0]  , fw_centers[:,1]  , 'ro')
              ax.plot(glob_fw_center[0], glob_fw_center[1], 'ks', markersize=7)
              ax.plot(glob_com[0]      , glob_com[1]      , 'b*', markersize=5)

              # Plot the original point
              ax.scatter(original_points[:,0], original_points[:,1],alpha=0.2)

              # Get the convex hull of the centers
              hull = ConvexHull(fw_centers)

              for simplex in hull.simplices:
                  plt.plot(fw_centers[simplex, 0], fw_centers[simplex, 1], 'r-',lw=1)

              plt.savefig('fermat_weber_plots/myplot_' + str(filenum).zfill(4) + '.png',
                          bbox_inches='tight', dpi=200)
 
              filenum += 1

              # Current fermat-weber centers are points for next iteration
              points = fw_centers


              #-----------------------------------------------------------------------------------
              ex_sum     = sum_dists_to_pt( original_points, glob_fw_center)
              my_apx_sum = sum_dists_to_pt( original_points, fw_centers[0])
              com_sum    = sum_dists_to_pt( original_points, glob_com)
          

              print Fore.YELLOW, "Exact FW center summation: ", ex_sum, Style.RESET_ALL
              print Fore.YELLOW, "My Approximate FW center summation: ", my_apx_sum, Style.RESET_ALL
              print Fore.GREEN, "Ratio: ", my_apx_sum/ex_sum, Style.RESET_ALL

              print "-----------------------------------------------------------------------------"
              print Fore.CYAN, "Exact FW center summation: ", ex_sum, Style.RESET_ALL
              print Fore.CYAN, "Center of mass center summation: ", com_sum, Style.RESET_ALL
              print Fore.CYAN, "Ratio: ", com_sum/ex_sum, Style.RESET_ALL


          
         



# Some basic canvas functions
def applyAxCorrection(ax):
      ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
      ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
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


# Implementation of the key-press handler

def wrapperkeyPressHandler(fig,ax, run): 
      def _keyPressHandler(event):
                      
             if event.key in ['c', 'C']: 
                   # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    utils_graphics.applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
             
             elif event.key in ['d','D']: 
                  run.runRecDel(fig, ax)

             elif event.key in ['m', 'M']:
                 run.maxDispDelTri(fig,ax)


             elif event.key in ['n', 'N', 'u', 'U']: 
                    # Generate a bunch of uniform or non-uniform random points on the canvas
                    numpts = int(raw_input("\n" + Fore.YELLOW+\
                                           "How many points should I generate?: "+\
                                           Style.RESET_ALL)) 
                    run.clearAllStates()
                    ax.cla()
                                   
                    utils_graphics.applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                    
                    fig.texts = []
                                     
                    import scipy
                    if event.key in ['n', 'N']: 
                            run.sites = utils_algo.bunch_of_non_uniform_random_points(numpts)
                    else : 
                            run.sites = scipy.rand(numpts,2).tolist()

                    patchSize  = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/140.0

                    for site in run.sites:      
                        ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                     facecolor='blue',edgecolor='black' ))

                    ax.set_title('Points : ' + str(len(run.sites)), fontdict={'fontsize':40})
                    fig.canvas.draw()





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
                 patchSize  = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/140.0
                    
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='blue', edgecolor='black'  ))
                 ax.set_title('Number of sites : ' + str(len(run.sites)), \
                              fontdict={'fontsize':40})
                 

             # Clear polygon patches and set up last minute \verb|ax| tweaks
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPoints

if __name__ == "__main__":
    # Body of main function
    
    fig, ax =  plt.subplots()
    run = HorseflyInputGraph()
            
    ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
    ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
              
    mouseClick = wrapperEnterPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
             
    keyPress   = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress )
        
    plt.show()
    
