# Relevant imports
from colorama import Fore, Style
from matplotlib import rc
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import inspect 
import itertools
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.style.use('seaborn-poster')
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time
import utils_algo
import utils_graphics

import problem_classic_horsefly as chf

# Set up logging information relevant to this module
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def debug(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.debug('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))

def info(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.info('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))

def run_handler():
    # Define key-press handler
       
    # The key-stack argument is mutable! I am using this hack to my advantage.
    def wrapperkeyPressHandler(fig,ax, run): 
           def _keyPressHandler(event):
               if event.key in ['i', 'I']:  
                    # Start entering input from the command-line
                    # Set speed and number of flies
                    
                    phi_str = raw_input(Fore.YELLOW + "What should I set the speed of each of the flies to be (should be >1)? : " + Style.RESET_ALL)
                    nof_str = raw_input(Fore.YELLOW + "How many flies do you want me to assign to the horse? : " + Style.RESET_ALL)

                    phi = float(phi_str)
                    nof = int(nof_str)
                    
                    # Select algorithm to execute

                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (ec)    Earliest Capture \n"                                        +\
                            " (ecp)   Earliest Capture Postopt   \n"                                         +\
                            " (tsp)   TSP order Postopt   \n"                                         +\
                            " (sp)    Split Partition  \n"                                        +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    utils_graphics.clearAxPolygonPatches(ax)

                    if   algo_str == 'ec':
                          tour = run.getTour(algo_greedy_earliest_capture, phi, \
                                             number_of_flies = nof)
                    elif algo_str == 'sp':
                          tour = run.getTour(algo_split_partition, phi, \
                                             number_of_flies = nof)   

                    elif algo_str == 'ecp':
                          tour = run.getTour(algo_earliest_capture_postopt , phi, \
                                             number_of_flies = nof)   

                    elif algo_str == 'tsp':
                          tour = run.getTour(algo_tsp_postopt , phi, \
                                             number_of_flies = nof)   
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()

                    utils_graphics.applyAxCorrection(ax)
                    plot_tour(ax, tour)
                    fig.canvas.draw()
                    
                    
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
                    
               elif event.key in ['c', 'C']: 
                    # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    utils_graphics.applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
                    
           return _keyPressHandler
    
    # Set up interactive canvas
    fig, ax =  plt.subplots()
    run = MultipleFliesInput()
    #print run
        
    ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
    ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    mouseClick   = utils_graphics.wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
    

# Local data-structures
class MultipleFliesInput:
      def __init__(self, sites=[], inithorseposn=()):
           self.sites           = sites
           self.inithorseposn   = inithorseposn

      # Methods for \verb|MultipleFliesInput|
      def clearAllStates (self):
         self.sites = []
         self.inithorseposn = ()
      def getTour(self, algo, speedratio, number_of_flies):
            return algo(self.sites, self.inithorseposn, speedratio, number_of_flies)
      


# Algorithms for multiple flies

# Helper functions for \verb|algo_greedy_earliest_capture|
   
def meeting_time_horse_fly_opp_dir(horseposn, flyposn, flyspeed):
    horseposn = np.asarray(horseposn)
    flyposn   = np.asarray(flyposn)
    return 1/(flyspeed+1) * np.linalg.norm(horseposn-flyposn)
    
# Definition of the \verb|FlyState| class
class FlyState:
    def __init__(self, idx, initflyposn, site, flyspeed):

         self.idx                                = idx
         self._flytraj                           = [ {'coordinates': np.asarray(initflyposn), 'type':'gen_pt'} ]
         self._current_assigned_site             = np.asarray(site)
         self._speed                             = flyspeed
         self._current_assigned_site_serviced_p  = False
         self._fly_retired_p                     = False
    
    def retire_fly(self):
         self._fly_retired_p = True
 
    def deploy_to_site(self,site):
         self._current_assigned_site            = np.asarray(site)
         self._current_assigned_site_serviced_p = False 

    def is_retired(self):
         return self._fly_retired_p

    def is_current_assigned_site_serviced(self):
         return self._current_assigned_site_serviced_p

    def get_current_fly_position(self):
         return self._flytraj[-1]['coordinates']
   
    def get_trajectory(self):
         return self._flytraj

    # Definition of method \verb|update_fly_trajectory|
    
    def update_fly_trajectory(self, dt, rendezvous_pt):

         if self.is_retired():
            return 

         dx = self._speed * dt

         if self._current_assigned_site_serviced_p :
            # Move towards the provided rendezvous point
               
            heading  = rendezvous_pt - self.get_current_fly_position()
            uheading = heading / np.linalg.norm(heading) 
            newpt    = self.get_current_fly_position() + dx * uheading
            self._flytraj.append(  {'coordinates': newpt, 'type': 'gen_pt'}  )
            

         elif dx < np.linalg.norm(self._current_assigned_site - self.get_current_fly_position()) :
            # Continue moving towards the site
            
            heading  = self._current_assigned_site - self.get_current_fly_position()
            uheading = heading / np.linalg.norm(heading) 
            newpt    = self.get_current_fly_position() + dx * uheading
            self._flytraj.append(  {'coordinates': newpt, 'type': 'gen_pt'}  )
            
         else: 
            # Move towards the site mark site as serviced and then head towards rendezvous point
            
            dx_reduced = dx - np.linalg.norm(self._current_assigned_site -\
                                             self.get_current_fly_position())
            heading  = rendezvous_pt - self._current_assigned_site
            uheading = heading/np.linalg.norm(heading)

            newpt = self._current_assigned_site + uheading * dx_reduced
            self._current_assigned_site_serviced_p = True
            self._flytraj.extend([{'coordinates':self._current_assigned_site, 'type':'site'}, 
                                  {'coordinates':newpt,                       'type':'gen_pt'}])
            
     
    # Definition of method \verb|rendezvous_time_and_point_if_selected_by_horse|
    
    def rendezvous_time_and_point_if_selected_by_horse(self, horseposn):
       assert(self._fly_retired_p != True)
      
       if self._current_assigned_site_serviced_p:
           rt = meeting_time_horse_fly_opp_dir(horseposn, self.get_current_fly_position(), self._speed)
           horseheading = self.get_current_fly_position() - horseposn
       else:
          distance_to_site    = np.linalg.norm(self.get_current_fly_position() -\
                                               self._current_assigned_site)
          time_of_fly_to_site = 1/self._speed * distance_to_site

          horse_site_vec   = self._current_assigned_site - horseposn 
          displacement_vec = time_of_fly_to_site * horse_site_vec/np.linalg.norm(horse_site_vec)
          horseposn_tmp   = horseposn + displacement_vec

          time_of_fly_from_site = \
                   meeting_time_horse_fly_opp_dir(horseposn_tmp, self._current_assigned_site, self._speed)

          rt = time_of_fly_to_site + time_of_fly_from_site
          horseheading = self._current_assigned_site - horseposn

       uhorseheading = horseheading/np.linalg.norm(horseheading)
       return rt, horseposn + uhorseheading * rt


#--------------------------------------------------------------------------------
#  ___      _ _ _     ___          _   _ _   _          
# / __|_ __| (_) |_  | _ \__ _ _ _| |_(_) |_(_)___ _ _  
# \__ \ '_ \ | |  _| |  _/ _` | '_|  _| |  _| / _ \ ' \ 
# |___/ .__/_|_|\__| |_| \__,_|_|  \__|_|\__|_\___/_||_|
#     |_|  
#--------------------------------------------------------------------------------
def algo_split_partition(sites, inithorseposn, phi, number_of_flies,\
                         write_algo_states_to_disk_p = False,      \
                         write_io_p                  = False,     \
                         animate_tour_p              = False):
      # Set algo-state and input-output files config
      import sys, datetime, os, errno
      import networkx as nx
      import itertools
      import doubling_tree as dbtree

      algo_name    = 'algo-split-partition'
      time_stamp   = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
      iter_counter = 0 

      if write_algo_states_to_disk_p or write_io_p or animate_tour_p:
        dir_name     = algo_name + '---' + time_stamp
        io_file_name = 'input_and_output.yml'
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

      sptree = algo_split_partition_tree(sites, inithorseposn,phi,number_of_flies,
                                         write_algo_states_to_disk_p,
                                         write_io_p,
                                         animate_tour_p)

      db_sptree        = dbtree.double_edges_of_graph(sptree)          ; #print Fore.RED   ; utils_algo.print_list(db_phi_mst.edges.data()) 
      shortcutted_tour = dbtree.get_shortcutted_euler_tour(db_sptree, source=len(sites))
      leaves           = [sptree.nodes[nidx]['position'] for nidx in shortcutted_tour if sptree.nodes[nidx]['type'] == 'leaf']

      assert len(leaves) == len(sites), "The set of leaves so extracted must be some permutation of the sites"
      assert np.linalg.norm(inithorseposn-sptree.nodes[len(sites)]['position']) < 1e-7, "The zeroth node must be the source"

      from itertools import cycle
      dronecycle        = cycle(range(number_of_flies)) 
      collection_info_1 = [ {'drone_collected'    : None, 
                          'returning_from_site' : None}]
      collection_info_2 = [ {'drone_collected'    : elt[0], 
                           'returning_from_site': elt[1]} for elt in zip(dronecycle, leaves)]
      collection_info   = collection_info_1 + collection_info_2

      assert len(collection_info)-1 == len(sites), \
       "The length of collections info should be exactly once less than the number of sites, because of initial point"

      horse_traj, fly_trajs = algo_exact_given_specific_ordering(inithorseposn, phi, number_of_flies, collection_info)

      print Fore.RED; utils_algo.print_list(horse_traj) ; print Style.RESET_ALL
      
      for ftraj, fidx in zip(fly_trajs, range(number_of_flies)):
          print Fore.GREEN, "\n Fly Traj ", fidx, Style.RESET_ALL
          utils_algo.print_list(ftraj)


      if write_io_p:
            data = { 'sites'            : sites,        \
                     'inithorseposn'    : inithorseposn,\
                     'phi'              : phi,          \
                     'horse_trajectory' : horse_traj,   \
                     'fly_trajectories' : fly_trajs }
            utils_algo.write_to_yaml_file(data, dir_name = dir_name, file_name = io_file_name)

      if animate_tour_p:
           animate_tour(sites            = sites, 
                        inithorseposn    = inithorseposn, 
                        phi              = phi, 
                        horse_trajectory = horse_traj, 
                        fly_trajectories = fly_trajs,
                        animation_file_name_prefix = dir_name + '/' + io_file_name)

      # Return multiple flies tour
      return {'sites'            : sites, \
               'inithorseposn'   : inithorseposn,\
               'phi'             : phi,\
               'horse_trajectory': horse_traj, \
               'fly_trajectories': fly_trajs}
 





def algo_split_partition_tree(sites, inithorseposn, phi, number_of_flies,\
                              write_algo_states_to_disk_p = False,       \
                              write_io_p                  = False,       \
                              animate_tour_p              = False) :

    def get_line_between_two_points(pt0, pt1):
        """ Line is represented as y = m*x+c """
        coefficients = np.polyfit( [pt0[0], pt1[0]] ,  [pt0[1], pt1[1]] , 1)
        m, c         =  coefficients
        return m, c
    
    def same_side_on_line_as_point(line, refpt, querypt):
        """If a point is in one of the open half-planes and the other is 
           on the line, they are counted to be on the same side of the line """

        m, c = line
        [x0,y0], [x1,y1] = refpt, querypt

        if (y0-m*x0-c)*(y1-m*x1-c) >= 0:
            return True
        else: # the points lie in the opposite open 
              # half-planes determined by the line
            return False

    def split_points_with_line( leafpts_info, spinept, line ):
        """ For a given bunch of points (the leafpts) and a reference point
            split the leafpoints into two sets. The first set returned consists 
            of points lies on the same side as that of the spinept. """
        idxA, idxB     = [ ] , [ ] 
        for r, g, pt in leafpts_info:
            if same_side_on_line_as_point(line, spinept, pt):
                idxA.append( (r,g)  )
            else:
                idxB.append( (r,g) )
        return idxA, idxB


    def get_center(points, branchpt, branchpt_weight):
        """ Get center of mass or fermat-weber center """
        #assert(branchpt_weight > 0.0)
        points             = map(np.asarray, points)
        branchpt           = np.asarray(branchpt)
        numcopies_branchpt = int(branchpt_weight)
        newpoints          = points + [branchpt for _ in range(numcopies_branchpt)]  
        xc, yc             = 0.0, 0.0

        for i  in range(len(newpoints)):
            xc += newpoints[i][0]
            yc += newpoints[i][1]

        xc = xc/len(newpoints)
        yc = yc/len(newpoints)
        return np.asarray([xc,yc])
 
    def leaf_degree(G, nidx):
        nbrs    = G[nidx]
        leafdeg = 0
        for i in nbrs:
            if G.nodes[i]['type']  == 'leaf':
               leafdeg += 1
        return leafdeg

    def draw_graph(G, inithorseposn, itercounter, outdir):
        """ Draw the state of the split partition graph """
        fig, ax =  plt.subplots()
        ax.set_xlim([utils_graphics.xlim[0], utils_graphics.xlim[1]])
        ax.set_ylim([utils_graphics.ylim[0], utils_graphics.ylim[1]])
        ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Progress of the split partition algorithm: Step '    + str(itercounter) +\
                     '\n  Speed: '+ str(phi) + ' Number of Drones ' + str(number_of_flies ), 
                     fontname='Times New Roman')
        filename = outdir+'/plot_'+ str(itercounter).zfill(5) +'.png'
        #----------------    
        # Draw edges
        #----------------    
        
        edges_G = G.edges()
        segs    = []
        for u,v in edges_G:
            px,py = G.nodes[u]['position']
            qx,qy = G.nodes[v]['position']

            if G.nodes[u]['type'] == 'leaf' or G.nodes[v]['type'] == 'leaf':
                plt.plot([px,qx],[py,qy],"b-", zorder=1)
            else :
                plt.plot([px,qx],[py,qy],"r-", zorder=1)
            #----------------    
            # Render node u
            #----------------    
            if G.nodes[u]['type'] == 'spine':
                ax.add_patch(mpl.patches.Circle((px,py), radius=1.0/100, facecolor='red', edgecolor='black',zorder=2))
                #ax.text(px, py, str(u), horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='#ff5733', edgecolor = '#ff5733', alpha=1.0), zorder=3) 
            else :
                ax.add_patch(mpl.patches.Circle((px,py), radius=1.0/120, facecolor='blue', edgecolor='black',zorder=2))
                #ax.text(px, py, str(u), horizontalalignment='center', verticalalignment='center' , bbox=dict(facecolor='#3355ff',  edgecolor = '#3355ff', alpha=1.0), zorder=3) 
            #----------------    
            # Render node v
            #----------------    
            if G.nodes[v]['type'] == 'spine':
                ax.add_patch(mpl.patches.Circle((qx,qy), radius=1.0/100, facecolor='red', edgecolor='black',zorder=2))
                #ax.text(qx, qy, str(v), horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='#ff5733',  edgecolor = '#ff5733',alpha=1.0), zorder=3)
            else :
                ax.add_patch(mpl.patches.Circle((qx,qy), radius=1.0/120, facecolor='blue', edgecolor='black',zorder=2))
                #ax.text(qx, qy, str(v), horizontalalignment='center', verticalalignment='center' , bbox=dict(facecolor='#3355ff', edgecolor = '#3355ff', alpha=1.0), zorder=3) 
        
        # Draw initial position of the horse
        ax.add_patch(mpl.patches.Circle((inithorseposn[0],inithorseposn[1]), radius=1.0/90, facecolor='green', edgecolor='black',zorder=3))

        plt.savefig(filename,dpi=400,bbox_inches='tight')
        plt.close('all')

    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    import networkx as nx
    import itertools
    algo_name    = 'algo-split-partition-tree'
    time_stamp   = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    iter_counter = 0 

    if write_algo_states_to_disk_p or write_io_p:
        dir_name     = algo_name + '---' + time_stamp
        io_file_name = 'input_and_output.yml'
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Using more flies than the number of sites, does not result 
    # in any speedup. So cut down the number of flies to just the
    # number of sites
    numsites = len(sites)
    if number_of_flies > numsites: 
          number_of_flies = numsites

    inithorseposn = np.asarray(inithorseposn)
    sites         = map(np.asarray, sites)
    G             = nx.Graph()
    node_attrs    = {}

    for site, idx in zip(sites, range(numsites)):
        node_attrs[idx] = {'type': 'leaf', 'position': site}  
        G.add_edge(numsites, idx, weight = np.linalg.norm(site-inithorseposn))

    node_attrs[numsites] = {'type':'spine', 'position': inithorseposn}    
    nx.set_node_attributes(G, node_attrs)
    high_degree_nodes = [n for n,v in G.nodes(data=True) 
                         if v['type'] == 'spine' and leaf_degree(G,n) > number_of_flies ]

    assert len(high_degree_nodes) == 1, "At this point there should be exactly one high degree node, the initial position of the truck"

    # Exactly one high-degree node is processed in each round of the while loop
    # More high-degree nodes maybe added during the while loop.
    while high_degree_nodes :

        print "\n------------------------------------\n",  \
              "ITERATION NUMBER: ", iter_counter, \
              "\n------------------------------------\n"
        print Fore.GREEN, "Split Partition Tree" , Style.RESET_ALL
        utils_algo.print_list(G.nodes(data=True))
        utils_algo.print_list(G.edges(data=True))
        print Fore.RED, " High Degree Nodes              :  ", high_degree_nodes, Style.RESET_ALL
        print Fore.RED, " High Degree Nodes  leaf degrees:  ", [leaf_degree(G,n) for n in high_degree_nodes], Style.RESET_ALL

        if write_algo_states_to_disk_p:
            draw_graph(G, inithorseposn,  iter_counter, dir_name )

        spinept_idx  = high_degree_nodes[0] 
        spinept      = G.nodes[spinept_idx]['position']
        leafpts      = G[spinept_idx]  
        leafpts_info = []
        ctr          = 0
        for nidx in leafpts:
            if G.nodes[nidx]['type'] == 'leaf':
                leafpts_info.append( (ctr, nidx, G.nodes[nidx]['position']) )
                ctr += 1

        leafpts_coods      = [p for _, _, p in leafpts_info ]
        idxA_opt, idxB_opt = [], [] 
        wt_opt             = np.inf

        for ptpair in itertools.combinations(leafpts_info, 2):
            (_, _, p) = ptpair[0]
            (_, _, q) = ptpair[1]
            [m,c]         = get_line_between_two_points(p,q)
            dtheta        = 0.01
            alpha, beta   = (p+q)/2.0
            
            for th in [dtheta, -dtheta]:
                # y = mstar * x + cstar is the line passint through (alpha, beta) 
                # rotated about y=mx+c by small angle theta
                mstar = (m+np.tan(th))/(1-m*np.tan(th))
                cstar = beta - mstar * alpha

                [idxA, idxB]  = split_points_with_line( leafpts_info, spinept, line=(mstar,cstar) )
                assert idxB, "The array idxB should not be empty"
                center        = get_center( [leafpts_coods[r] for (r,_) in idxB] , spinept, phi )

                wt = 2.0 * sum ([ np.linalg.norm(spinept-leafpts_coods[r]) for (r,_) in idxA] ) +\
                     phi * np.linalg.norm(spinept-center)                                        +\
                     2.0 * sum ([ np.linalg.norm(center-leafpts_coods[r])  for (r,_) in idxB] )

                #--------------------------
                # Comparison code for opt
                #--------------------------
                if wt < wt_opt:
                    print Fore.GREEN, " Better Line Partition Found! ", Style.RESET_ALL
                    #print Fore.GREEN, " idxA = ", idxA, Style.RESET_ALL
                    #print Fore.GREEN, " idxB = ", idxB, Style.RESET_ALL
                    wt_opt             = wt
                    idxA_opt, idxB_opt = idxA, idxB
                    center_opt         = center

        if (not idxB_opt):
            print "\n idxB_opt detected as empty. This should not happen"
            print Fore.RED, "idxA_opt: ", idxA_opt 
            print Fore.RED, "idxB_opt: ", idxB_opt 
            sys.exit()

        #-------------------------------------------
        # Modify graph G after performing the split
        #-------------------------------------------
        # Delete edges 
        for (_,g) in idxB_opt:
            G.remove_edge(spinept_idx,g)

        newnodeidx = G.number_of_nodes()
        G.add_node(newnodeidx,type = 'spine', position = center_opt)
        G.add_edge(newnodeidx, spinept_idx, weight=np.linalg.norm(spinept - center_opt))

        for (r,g) in idxB_opt:
            edgewt = np.linalg.norm(center_opt-leafpts_coods[r])
            G.add_edge(g, newnodeidx, weight=edgewt)

        #------------------------------------
        # Modify the array highdegree nodes
        #------------------------------------
        if leaf_degree(G,spinept_idx) <= number_of_flies:
            high_degree_nodes.remove(spinept_idx)
            
        if leaf_degree(G, newnodeidx) > number_of_flies:
            high_degree_nodes.append(newnodeidx)

        #------------------------------------------
        # Write state of current iteration to disk
        #------------------------------------------
        if write_algo_states_to_disk_p:
            iter_state_file_name = 'iter_state_' + str(iter_counter).zfill(5) + '.yml'
            data                 = G
            utils_algo.write_to_yaml_file(data, dir_name=dir_name, file_name=iter_state_file_name)
        iter_counter += 1


    return G


   




#--------------------------------------------------------------------------------------------------------
def algo_greedy_earliest_capture(sites, inithorseposn, phi, number_of_flies,\
                                 write_algo_states_to_disk_p = False,\
                                 write_io_p                  = False,\
                                 animate_tour_p              = False) :

    # Set algo-state and input-output files config
    import sys, datetime, os, errno
    algo_name     = 'algo-greedy-earliest-capture'
    time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
    dir_name      = algo_name + '---' + time_stamp
    io_file_name  = 'input_and_output.yml'

    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    algo_state_counter = 1 
    

    if number_of_flies > len(sites):
          number_of_flies = len(sites)

    current_horse_posn = np.asarray(inithorseposn)
    horse_traj         = [(current_horse_posn, None)]

    # Find the $k$-nearest sites to \verb|inithorseposn| for $k=$\verb|number_of_flies| and claim them
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=number_of_flies)
    neigh.fit(sites)

    _, knn_idxss = neigh.kneighbors([inithorseposn])
    knn_idxs     = knn_idxss.tolist()[0]
    knns         = [sites[i] for i in knn_idxs]
    unclaimed_sites_idxs = list(set(range(len(sites))) - set(knn_idxs)) # https://stackoverflow.com/a/3462160
    
    # Initialize one \verb|FlyState| object per fly for all flies
    flystates = []
    for i in range(number_of_flies):
        flystates.append(FlyState(i,inithorseposn, knns[i], phi))
    

    all_flies_retired_p = False

    while (not all_flies_retired_p):
       # Find the index of the fly \bm{F} which can meet the horse at the earliest, the rendezvous point $R$, and time till rendezvous
       imin  = 0
       rtmin = np.inf
       rptmin= None
       for i in range(number_of_flies):
            if flystates[i].is_retired():
                continue
            else:
                rt, rpt = flystates[i].rendezvous_time_and_point_if_selected_by_horse(current_horse_posn)
                if rt < rtmin:
                    imin   = i
                    rtmin  = rt
                    rptmin = rpt
       
       # Update fly trajectory in each \verb|FlyState| object till \bm{F} meets the horse at $R$
       for flystate in flystates:
           flystate.update_fly_trajectory(rtmin, rptmin)
        
       # Update \verb|current_horse_posn| and horse trajectory
       current_horse_posn = rptmin
       horse_traj.append((np.asarray(rptmin),imin))
       
       # Deploy \bm{F} to an unclaimed site if one exists and claim that site, otherwise retire \bm{F}
         
       if  unclaimed_sites_idxs:
           unclaimed_sites = [sites[i] for i in unclaimed_sites_idxs]

           neigh = NearestNeighbors(n_neighbors=1)
           neigh.fit(unclaimed_sites)

           _, nn_idxss = neigh.kneighbors([current_horse_posn])
           nn_idx      = nn_idxss.tolist()[0][0]

           flystates[imin].deploy_to_site(unclaimed_sites[nn_idx])
           unclaimed_sites_idxs = list(set(unclaimed_sites_idxs) - \
                                       set([unclaimed_sites_idxs[nn_idx]]))

       else: 
           flystates[imin].retire_fly()
        
       # Calculate value of \verb|all_flies_retired_p|
       acc = True 
       for i in range(number_of_flies):
            acc = acc and flystates[i].is_retired()
       all_flies_retired_p = acc
       
       # Write algorithms current state to file, if \verb|write_algo_states_to_disk_p == True|
       
       print "Algorithm State Number: ", algo_state_counter
       if write_algo_states_to_disk_p:
            algo_state_file_name = 'algo_state_' + str(algo_state_counter).zfill(5) + '.yml'

            data = {'horse_trajectory' : horse_traj, \
                    'fly_trajectories' : [flystates[i].get_trajectory() for i in range(number_of_flies)] }
            utils_algo.write_to_yaml_file(data, dir_name=dir_name, file_name=algo_state_file_name)
       algo_state_counter += 1
        
    
    # Write input and output to file if \verb|write_io_p == True|
    if write_io_p:
         data = { 'sites' : sites, \
                  'inithorseposn' : inithorseposn,\
                  'phi':phi,\
                  'horse_trajectory' : horse_traj, \
                  'fly_trajectories' : [flystates[i].get_trajectory() 
                                       for i in range(number_of_flies)] }
         utils_algo.write_to_yaml_file(data, dir_name = dir_name, file_name = io_file_name)
    
    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(sites            = sites, 
                     inithorseposn    = inithorseposn, 
                     phi              = phi, 
                     horse_trajectory = horse_traj, 
                     fly_trajectories = [flystates[i].get_trajectory() for i in range(number_of_flies)],
                     animation_file_name_prefix = dir_name + '/' + io_file_name)
    
    # Return multiple flies tour
    return {'sites'           : sites,        \
            'inithorseposn'   : inithorseposn,\
            'phi'             : phi,          \
            'horse_trajectory': horse_traj,   \
            'fly_trajectories': [flystates[i].get_trajectory() for i in range(number_of_flies)]}
    

#-----------------------------------------------------------------------------------------
def algo_tsp_postopt (sites, inithorseposn, phi, number_of_flies,\
                      write_algo_states_to_disk_p = False,\
                      write_io_p                  = False,\
                      animate_tour_p              = False) :

    # Get the TSP order on the set of points 
    from concorde.tsp import TSPSolver
    import math

    inithorseposn = np.asarray(inithorseposn)
    sites         = map (np.asarray, sites)

    horseinit_and_sites = [inithorseposn] + sites

    utils_algo.print_list(horseinit_and_sites)
    
    # Concorde only accepts integer value coordinates. 
    # Hence I scale by a large factor (arbitrarily chosen as 1000), 
    # and then take a floor
    xs       = [ int(math.floor(1000*x)) for (x,_) in horseinit_and_sites ]
    ys       = [ int(math.floor(1000*y)) for (_,y) in horseinit_and_sites ]
    solution = TSPSolver.from_data(xs,ys,norm='EUC_2D').solve()
    
    assert solution.found_tour, "Did not find solution tour"
    tsp_idxs = list(solution.tour)
    h        = tsp_idxs.index(0) 
    tsp_idxs = tsp_idxs[h:] + tsp_idxs[:h]
    
    site_idxs     = [idx - 1 for idx in tsp_idxs[1:]] # because all site ids were shifted forward by 1
                                                      # when solving the tsp. here we retrieve them back
    sites_ordered = [sites[idx] for idx in site_idxs]

    from itertools import cycle
    dronecycle        = cycle(range(number_of_flies)) 
    collection_info_1 = [ {'drone_collected'    : None, 
                          'returning_from_site' : None}]
    collection_info_2 = [ {'drone_collected'    : elt[0], 
                           'returning_from_site': elt[1]} for elt in zip(dronecycle, sites_ordered)]
    collection_info   = collection_info_1 + collection_info_2
 

    assert len(collection_info)-1 == len(sites), \
    "The length of collections info should be exactly once less than the number of sites, because of initial point"

    horse_traj, fly_trajs = algo_exact_given_specific_ordering(inithorseposn, phi, number_of_flies, collection_info)
 

    return {'sites'           : sites,        \
            'inithorseposn'   : inithorseposn,\
            'phi'             : phi,          \
            'horse_trajectory': horse_traj,   \
            'fly_trajectories': fly_trajs     }
 

#-------------------------------------------------------------------------------------------------------
def algo_earliest_capture_postopt (sites, inithorseposn, phi, number_of_flies,\
                                   write_algo_states_to_disk_p = False,\
                                   write_io_p                  = False,\
                                   animate_tour_p              = False) :
    """
    Just run the earliest capture heuristic and record the order of capture. 
    Then having obtained the exact order in which the capture happens
    for each drone pass it onto the exact ordering SOCP solver. 
    """
    
    answer = algo_greedy_earliest_capture(sites, inithorseposn, phi, number_of_flies,\
                                write_algo_states_to_disk_p = False, \
                                write_io_p                  = False,\
                                animate_tour_p              = False)
 
    fly_trajectories        = answer['fly_trajectories']
    fly_trajectories_filter = [ [ pt['coordinates']  for pt in fly_trajectories[i] if pt['type'] == 'site' ]  
                                 for i in range(number_of_flies)   ]
    collection_info = []
    for elt in answer['horse_trajectory']:

        idx_drone_collected = elt[1]

        if idx_drone_collected is None:
            collection_info.append({ 'drone_collected'    : None,  
                                     'returning_from_site': None   })
        else:
            collection_info.append({'drone_collected'    : idx_drone_collected,
                                    'returning_from_site': fly_trajectories_filter[idx_drone_collected].pop(0)})

    # Check that all the filtered trajectories are now empty
    for traj in fly_trajectories_filter:
        assert len(traj) == 0 , "all fly_trajectories_filter should be empty"


    assert len(collection_info)-1 == len(sites), \
        "The length of collections info should be exactly once less than the number of sites, because of initial point"


    print Fore.RED, " Just before calling post opt", Style.RESET_ALL

    horse_traj, fly_trajs = algo_exact_given_specific_ordering(inithorseposn, phi, number_of_flies, collection_info)
  
        
    # Animate compute tour if \verb|animate_tour_p == True|
    if animate_tour_p:
        animate_tour(sites            = sites, 
                     inithorseposn    = inithorseposn, 
                     phi              = phi, 
                     horse_trajectory = horse_traj, 
                     fly_trajectories = fly_trajs,
                     animation_file_name_prefix = '')

    return {'sites'           : sites,        \
            'inithorseposn'   : inithorseposn,\
            'phi'             : phi,          \
            'horse_trajectory': horse_traj,   \
            'fly_trajectories': fly_trajs     }
    





#-------------------------------------------------------------------------------------------------
def algo_exact_given_specific_ordering(horseflyinit, phi, number_of_flies, collection_info):
    """ Using SOCP from Page 20 of
    https://pdfs.semanticscholar.org/23a4/3524fd5168acfd589e919c143f49a6eeeac3.pdf
    """
    import cvxpy as cp
    
    horseflyinit = np.asarray(horseflyinit)

    # Variables for rendezvous points of robot with package
    X = [cp.Variable(2)]
    t = [cp.Variable()] # associated with the initial position 

    for i in range( len(collection_info) ):
       X.append(cp.Variable(2)) # vector variable
       t.append(cp.Variable( )) # scalar variable

    # Constraints 
    constraints_start = [ X[0] == horseflyinit, \
                          t[0] == 0.0 ]

    #---------------------------------------------------------
    constraints_pos = [ ] 
    for i in range(1,len(collection_info)):
        constraints_pos.append( 0.0 <= t[i] )

    print Fore.GREEN, "Set pos constraints!!", Style.RESET_ALL
    #---------------------------------------------------------
    constraints_truck = []
    for i in range(len(collection_info)-1):
        constraints_truck.append( t[i] + cp.norm(X[i+1]-X[i]) <= t[i+1]  )

    print Fore.GREEN, "Set truck constraints!!", Style.RESET_ALL
    #---------------------------------------------------------
    constraints_drone = []
    
    drones_uncollected = range(number_of_flies)
    for q in range(1,len(collection_info)):
         idx_drone_collected = collection_info[q]['drone_collected']
         
         if idx_drone_collected in drones_uncollected:

             site = collection_info[q]['returning_from_site']
             f    = cp.norm(site - X[0])
             b    = cp.norm(site - X[q])
             constraints_drone.append( t[0] + (f+b)/phi <= t[q] ) 
             drones_uncollected.remove(idx_drone_collected)


    for q in range(len(collection_info)-1):
        idx_drone_collected_q = collection_info[q]['drone_collected']

        for r in range(q+1,len(collection_info)):
            idx_drone_collected_r = collection_info[r]['drone_collected']

            if idx_drone_collected_q == idx_drone_collected_r:

                site = collection_info[r]['returning_from_site']
                f    = cp.norm(site-X[q])
                b    = cp.norm(site-X[r])
                constraints_drone.append( t[q]+(f+b)/phi <= t[r] ) 
                break
    
    #---------------------------------------------------------
    objective = cp.Minimize(  t[len(collection_info)-1]  )
    prob      = cp.Problem(objective, constraints_start +\
                                      constraints_pos   +\
                                      constraints_drone +\
                                      constraints_truck)
    prob.solve(solver=cp.CVXOPT,verbose=True) 
    #---------------------------------------------------------
    horse_traj  = [ (np.asarray(X[i].value),  collection_info[i]['drone_collected'] ) 
                    for i in range(len(collection_info))]

    fly_trajs = [ [{ 'coordinates': X[0].value, 'type': 'gen_pt'}] for _ in range(number_of_flies)]
    for elt, k in zip(collection_info, range(len(collection_info))):
        if elt['drone_collected'] is None:
            pass
        else:
            didx_collected = elt['drone_collected']
            fly_trajs[didx_collected].append({'coordinates': elt['returning_from_site'] , 'type': 'site'  })
            fly_trajs[didx_collected].append({'coordinates': X[k].value                 , 'type': 'gen_pt'})

    return horse_traj, fly_trajs 
    
#--------------------------------------------------------------------------------------------------


# Plotting routines

def plot_tour(ax, tour):

    sites            = tour['sites']
    inithorseposn    = tour['inithorseposn']
    phi              = tour['phi']
    horse_trajectory = tour['horse_trajectory']
    fly_trajectories = tour['fly_trajectories']

    xhs = [ horse_trajectory[i][0][0] for i in range(len(horse_trajectory))]    
    yhs = [ horse_trajectory[i][0][1] for i in range(len(horse_trajectory))]    

    number_of_flies = len(fly_trajectories)
    colors          = utils_graphics.get_colors(number_of_flies, lightness=0.4)

    ax.cla()
    utils_graphics.applyAxCorrection(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot fly trajectories
    xfss = [[point['coordinates'][0] for point in fly_trajectories[i]] for i in range(len(fly_trajectories))]
    yfss = [[point['coordinates'][1] for point in fly_trajectories[i]] for i in range(len(fly_trajectories))]
 
    for xfs, yfs,i in zip(xfss,yfss,range(number_of_flies)):
        ax.plot(xfs,yfs,color=colors[i], alpha=0.7)

    # Plot sites along each flys tour
    xfsitess = [ [point['coordinates'][0] for point in fly_trajectories[i] if point['type'] == 'site'] 
                for i in range(len(fly_trajectories))]
    yfsitess = [ [point['coordinates'][1] for point in fly_trajectories[i] if point['type'] == 'site'] 
                for i in range(len(fly_trajectories))]
    
    for xfsites, yfsites, i in zip(xfsitess, yfsitess, range(number_of_flies)):
        for xsite, ysite, j in zip(xfsites, yfsites, range(len(xfsites))):
              ax.add_patch(mpl.patches.Circle((xsite,ysite), radius = 1.0/140, \
                                              facecolor=colors[i], edgecolor='black'))
              ax.text(xsite, ysite, str(j+1), horizontalalignment='center', 
                                              verticalalignment='center'  , 
                                              bbox=dict(facecolor=colors[i], alpha=1.0)) 
    # Plot horse tour
    ax.plot(xhs,yhs,'o-',markersize=5.0, linewidth=2.5, color='#D13131') 
    
    # Plot initial horseposn 
    ax.add_patch( mpl.patches.Circle( inithorseposn,radius = 1.0/100,
                                    facecolor= '#D13131', edgecolor='black'))


# Animation routines
   
def animate_tour (sites, inithorseposn, phi, horse_trajectory, fly_trajectories, animation_file_name_prefix):
    import numpy as np
    import matplotlib.animation as animation
    from   matplotlib.patches import Circle
    import matplotlib.pyplot as plt 

    # Set up configurations and parameters for all necessary graphics
       
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()

    # customize the major grid
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='red')

    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    # Visually distinct colors for displaying each flys trajectory in a different color 
    number_of_flies = len(fly_trajectories)
    colors          = utils_graphics.get_colors(number_of_flies, lightness=0.5)

    horse_trajectory_pts = map(lambda x: x[0], horse_trajectory)
    tour_length = utils_algo.length_polygonal_chain(horse_trajectory_pts)
        
    ax.set_title("Number of sites: " + str(len(sites)) + "\nTour Length: " + str(round(tour_length,4)), fontsize=15)
    ax.set_xlabel(r"$\varphi=$ " + str(phi) + "\nNumber of drones: " + str(number_of_flies), fontsize=15)
    
    # Parse trajectory information and convert trajectory representation to leg list form
     
    # Leg list form for all horse trajectories
    horse_traj_ll = []
    for i in range(len(horse_trajectory)-1):
        horse_traj_ll.append((horse_trajectory[i][0], horse_trajectory[i+1][0], 
                              horse_trajectory[i+1][1]))

    # Leg list form for all fly trajectories
    fly_trajs_ll = []
    for fly_traj in fly_trajectories:
        fly_traj_ll = []
        for i in range(len(fly_traj)-1):
            if fly_traj[i]['type'] == 'gen_pt':
     
                 if fly_traj[i+1]['type'] == 'gen_pt':
                      fly_traj_ll.append((fly_traj[i], 
                                          fly_traj[i+1]))
                      
        
                 elif fly_traj[i+1]['type'] == 'site':
                      fly_traj_ll.append((fly_traj[i], \
                                          fly_traj[i+1], \
                                          fly_traj[i+2]))
        fly_trajs_ll.append(fly_traj_ll)

    num_horse_legs = len(horse_traj_ll)

    # Append empty legs to fly trajectories so that leg counts 
    # for all fly trajectories are the same as that of the horse
    # trajectory
    for fly_traj in fly_trajs_ll:
        m = len(fly_traj)
        empty_legs = [None for i in range(num_horse_legs-len(fly_traj))]
        fly_traj.extend(empty_legs)

    
    # Construct and store every frame of the animation in the \verb|ims| array
       
    # Define discretization function for a leg of the horse or fly tour

    def discretize_leg(pts):
        subleg_pts = []

        if pts == None:
             return None
        else:
             numpts = len(pts)

             if numpts == 2:   # horse leg or fly-leg of type gg
                 k  = 19 
                 legtype = 'gg'
             elif numpts == 3: # fly leg of type gsg 
                 k  = 10 
                 legtype = 'gsg'

             pts = map(np.asarray, pts)
             for p,q in zip(pts, pts[1:]):
                 tmp = []
                 for t in np.linspace(0,1,k): 
                       tmp.append((1-t)*p + t*q) 
                 subleg_pts.extend(tmp[:-1])

             subleg_pts.append(pts[-1])
             return {'points': subleg_pts, 
                     'legtype'  : legtype}
    
    ims                = []
    horse_points_so_far = []
    fly_points_so_far   = [[] for i in range(number_of_flies)] 
    fly_sites_so_far    = [[] for i in range(number_of_flies)] # each list is definitely a sublist of corresponding list in fly_points_so_far
    for idx in range(len(horse_traj_ll)):
        # Get current horse-leg and update the list of points covered so far by the horse
        horse_leg = (horse_traj_ll[idx][0], horse_traj_ll[idx][1])
        horse_points_so_far.append(horse_leg[0]) # attach the beginning point of the horse leg
        horse_leg_pts = horse_leg
        #utils_algo.print_list(horse_points_so_far)
        #print "....................................................."

        fly_legs  = [fly_trajs_ll[i][idx] for i in range(len(fly_trajs_ll)) ]
        fly_legs_pts  = []
        for fly_leg, i in zip(fly_legs, range(len(fly_legs))):
           if fly_leg != None:
                coods = []
                for pt in fly_leg:
                     coods.append(pt['coordinates'])
                fly_legs_pts.append(coods)
                fly_points_so_far[i].append(coods[0]) # attaching the beginning point of the leg. Extension only 
                                                      # happens for legs wqhich are not of type None, meshing well 
                                                      # with the fact that fly has stopped moving. 
           else:
                fly_legs_pts.append(None)

        # discretize current leg, for horse and fly, and for each point in the discretization
        # render the frame. If a fly crosses a site, update the fly_points_so_far list
        horse_leg_disc = discretize_leg(horse_leg_pts)   # list of points 
        fly_legs_disc  = map(discretize_leg, fly_legs_pts) # list of list of points 

        # Each iteration of the following loop tacks on a new frame to ims
        # this outer level for loop is just proceeding through each position
        # in the discretized horse legs. This is the motion which coordinates
        # the flys motions
        for k in range(len(horse_leg_disc['points'])):
            current_horse_posn = horse_leg_disc['points'][k]
            current_fly_posns  = []  # updated in the for loop below.
            for j in range(len(fly_legs_disc)):
                  if fly_legs_disc[j] != None:
                        current_fly_posns.append(fly_legs_disc[j]['points'][k])

                        if fly_legs_disc[j]['legtype'] == 'gsg' and k==9: # yay, we just hit a site!
                              fly_points_so_far[j].append(fly_legs_disc[j]['points'][k])
                              fly_sites_so_far[j].append(fly_legs_disc[j]['points'][k])
                  else: 
                        current_fly_posns.append(None)
            objs = []

            # Plot trajectory of flies 
            assert(len(fly_points_so_far) == number_of_flies)
            for ptraj, i in zip(fly_points_so_far, range(number_of_flies)):
                 print current_fly_posns[i]
                 if current_fly_posns[i] is None:
                       xfs = [pt[0] for pt in ptraj]
                       yfs = [pt[1] for pt in ptraj] 
        
                 else:
                       xfs = [pt[0] for pt in ptraj] + [current_fly_posns[i][0]]
                       yfs = [pt[1] for pt in ptraj] + [current_fly_posns[i][1]]

                 flyline, = ax.plot(xfs, yfs, '-', linewidth=2.5, alpha=0.30, color=colors[i])
                 flyloc   = Circle((xfs[-1], yfs[-1]), 0.01, facecolor = colors[i], alpha=0.7)
                 flypatch = ax.add_patch(flyloc)
                 objs.append(flypatch)
                 objs.append(flyline)


            # Plot trajectory of horse
            xhs = [pt[0] for pt in horse_points_so_far] + [current_horse_posn[0]]
            yhs = [pt[1] for pt in horse_points_so_far] + [current_horse_posn[1]]
            horseline, = ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')
            horseloc   = Circle((current_horse_posn[0], current_horse_posn[1]), 0.02, facecolor = '#D13131', alpha=1.00)
            horsepatch = ax.add_patch(horseloc)
            objs.append(horseline)
            objs.append(horsepatch)

            # Plot sites as black circles
            for site in sites:
                 circle = Circle((site[0], site[1]), 0.01, \
                                  facecolor = 'k'   , \
                                  edgecolor = 'black'     , \
                                  linewidth=1.0)
                 sitepatch = ax.add_patch(circle)
                 objs.append(sitepatch)


            # Plot currently covered sites as colored circles
            for sitelist, i in zip(fly_sites_so_far, range(number_of_flies)):
               for site in sitelist:
                     circle = Circle((site[0], site[1]), 0.015, \
                                      facecolor = colors[i]   , \
                                      edgecolor = 'black'     , \
                                      linewidth=1.0)
                     sitepatch = ax.add_patch(circle)
                     objs.append(sitepatch)

            debug(Fore.CYAN + "Appending to ims "+ Style.RESET_ALL)
            ims.append(objs) 
    
    # Write animation of tour to disk and display in live window
       
    from colorama import Back 

    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=150)
    #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

    plt.show() # For displaying the animation in a live window. 
