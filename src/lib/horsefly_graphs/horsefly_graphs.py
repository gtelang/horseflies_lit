    
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import scipy as sp
import sys, os, time
import itertools
from colorama import Fore, Style
import logging
import utils_graphics, utils_algo
import networkx as nx
from CGAL.CGAL_Kernel import Point_2, Segment_2, Iso_rectangle_2
from CGAL.CGAL_Kernel import do_intersect, intersection

# Local data-structures

 
class HorseflyInputGraph:
      def __init__(self, sites=[], inithorseposn=[]):
           self.sites                = sites
           self.inithorseposn        = inithorseposn
           self.obstacle_list        = []
           self.horsefly_input_graph = nx.Graph()
    
           # Tracking variables used only during interactive input. 
           # To be frank these tracking variables should be placed in a separate class
           # they seem out of place here. Else have a dedicated class stored as a 
           # variable/dictionary containinig all these icky state variables used 
           # during input manipulation. 
           self.default_obstacle_width   = 0.1 
           self.default_obstacle_height  = 0.1 

           self.hovering_obstacle_width  = 0.1 # This will be modified during the move
           self.hovering_obstacle_height = 0.1 # This will be modified during the move

           self.horse_obstacle_input_mode_p = False
           self.fly_obstacle_input_mode_p = False
           self.common_obstacle_input_mode_p = False


      def clearAllStates (self):
          self.sites           = []
          self.inithorseposn   = None
          self.obstacle_list = [] 
   
          self.horse_obstacle_input_mode_p  = False
          self.fly_obstacle_input_mode_p    = False
          self.common_obstacle_input_mode_p = False
    
          self.horsefly_input_graph = nx.Graph()
      
      # Definition of method \verb|makeHorseflyInputGraph|
         
      def makeHorseflyInputGraph(self, fig, ax, background_grid_pts):
          # Create an initial list of nodes to be inserted into \verb|self.horsefly_input_graph|
             
          node_list = []

          # Insert background_grid_pts
          for pt in background_grid_pts:
                  node_list.append( {'coordinates': pt, 
                                     'point_type' : 'background'} )
          # Insert site points
          for site in run.sites:
                  node_list.append({'coordinates':site, 
                                    'point_type': 'site'})

          # Insert inithorseposn if it has been provided
          if run.inithorseposn:
                   node_list.append({'coordinates':run.inithorseposn, 
                                     'point_type':'inithorseposn'})
          
          # Extract data from the Delaunay Triangulation of the points corresponding to the nodes
          
          from scipy.spatial import Delaunay
          tri = Delaunay([node['coordinates'] for node in node_list])
          numtris = len(tri.simplices)

          #print tri.simplices

          del_tri_edge_list = []
          for simplex, fidx in zip(tri.simplices,range(numtris)):
                  [i,j,k] = simplex
                  
                  del_tri_edge_list.append([[min(i,j), max(i,j)], fidx])
                  del_tri_edge_list.append([[min(j,k), max(j,k)], fidx])
                  del_tri_edge_list.append([[min(k,i), max(k,i)], fidx])

          del_tri_edge_list.sort()
          from itertools import groupby
          del_tri_edge_list = [elt[0] for elt in groupby(del_tri_edge_list)]
           
          #utils_algo.print_list(del_tri_edge_list)
             
          
          # Add points along each edge of a simplex in the triangulation
             
          def add_points_interior_to_segment(p,q,k):
              p, q = map(np.asarray, [p,q])
              #return []
              return [p + float(i)/3.0 * (q-p) for i in [1,2]] 
           
          old_len_of_node_list = len(node_list)
          edges_processed      = {}
          face_info            = { fidx: [] for fidx in range(numtris)  }

          for [[i,j], fidx] in del_tri_edge_list:

                if (i,j) not in edges_processed.keys():
                        new_segment_pts = add_points_interior_to_segment(node_list[i]['coordinates'], 
                                                                 node_list[j]['coordinates'], k)
                        for pt in new_segment_pts:
                            node_list.append({'coordinates': pt, 'point_type' : 'background'})
              
                        num_new_nodes = len(new_segment_pts)

                        #print Fore.RED, "New nodes added: ", range(old_len_of_node_list, old_len_of_node_list + num_new_nodes), Style.RESET_ALL

                        edges_processed[(i,j)] = range(old_len_of_node_list, 
                                                       old_len_of_node_list+num_new_nodes)
                        face_info[fidx].append( ((i,j), edges_processed[(i,j)]) )  
                        old_len_of_node_list = len(node_list)
                else: 
                        face_info[fidx].append( ((i,j), edges_processed[(i,j)]) )  
                       
          
          # Iterate through every simplex and draw the complete graph on the points on the edges of simplex
             
              
          possible_edge_list_idxs = []
          for fidx, finfo in face_info.items(): 

              # Create graph edges along each simplex edge
              # and add to possible_edge_list_idxs
              for ((i,j), new_pts_idxs) in finfo:
                   tmp = [i] + new_pts_idxs + [j]
                   possible_edge_list_idxs.extend(zip(tmp,tmp[1:]))

              # Create graph edges interior to each face between 
              # every pair of edges and add to possible_edge_list_idxs
              [ (_, idxs1 ), (_, idxs2), (_, idxs3) ]= finfo
              
              #print Fore.YELLOW, finfo, Style.RESET_ALL

              possible_edge_list_idxs.extend([ (i,j) for i in idxs1 for j in idxs2 ])
              possible_edge_list_idxs.extend([ (i,j) for i in idxs2 for j in idxs3 ])
              possible_edge_list_idxs.extend([ (i,j) for i in idxs3 for j in idxs1 ])


          # Extract the actual nodes from node_list correponding to the 
          # the indexes in possible_edge_list_idxs
          possible_edge_list = map(lambda xs : [node_list[xs[0]], node_list[xs[1]]], 
                                   possible_edge_list_idxs)
          
          # Make a list of possible edges that will be tested for intersection against the obstacles
             

          for node,i in zip(node_list, range(len(node_list))):
              node['idx'] = i

          for pt_edge in possible_edge_list:

                       [p,q] = pt_edge
                       pcd = p['coordinates']
                       qcd = q['coordinates']
                       
                       obstacle_interscn_info = []
                       for obs in run.obstacle_list:
                             (interscn_p, _) = obs.intersectionWithSegment(pcd,qcd)
                             if interscn_p:
                                 obstacle_interscn_info.append(obs.obstype)    
           
                       # Make a count of the type of each obstacle intersected by segment
                       obs_interscn_count = {'horseobs': 0, 'flyobs':0, 'commonobs':0}
                       for obstype in obstacle_interscn_info:
                          if   obstype == 'horseobs':
                                   obs_interscn_count['horseobs']  += 1
                           
                          elif obstype == 'flyobs':
                                   obs_interscn_count['flyobs']    += 1

                          elif obstype == 'commonobs':
                                   obs_interscn_count['commonobs'] += 1


                       pcd = np.asarray(pcd) 
                       qcd = np.asarray(qcd)
                       
                       if   obs_interscn_count['horseobs']   == 0  and \
                          obs_interscn_count['flyobs']     == 0  and \
                          obs_interscn_count['commonobs']  == 0:

                          run.horsefly_input_graph.add_edge(p['idx'], q['idx'], edgetype='commonedge', weight=np.linalg.norm(qcd-pcd))
                          # modify the node attributes at both ends
                          run.horsefly_input_graph.nodes[p['idx']]['point_type'] = p['point_type']
                          run.horsefly_input_graph.nodes[p['idx']]['coordinates'] = p['coordinates']
                          
                          run.horsefly_input_graph.nodes[q['idx']]['point_type'] = q['point_type']
                          run.horsefly_input_graph.nodes[q['idx']]['coordinates'] = q['coordinates']


                          ax.plot( [pcd[0],qcd[0]], [pcd[1],qcd[1]], 'y-',  alpha=0.2 ) 


                       elif obs_interscn_count['horseobs']  >= 1  and \
                          obs_interscn_count['flyobs']      == 0  and \
                          obs_interscn_count['commonobs']   == 0:

                          run.horsefly_input_graph.add_edge(p['idx'], q['idx'], edgetype='flyedge', weight=np.linalg.norm(qcd-pcd))
                          # modify the node attributes at both ends
                          run.horsefly_input_graph.nodes[p['idx']]['point_type'] = p['point_type']
                          run.horsefly_input_graph.nodes[p['idx']]['coordinates'] = p['coordinates']
                          
                          run.horsefly_input_graph.nodes[q['idx']]['point_type'] = q['point_type']
                          run.horsefly_input_graph.nodes[q['idx']]['coordinates'] = q['coordinates']


                          ax.plot( [pcd[0],qcd[0]], [pcd[1],qcd[1]], 'b-',  alpha=0.2 ) 

              
                       elif obs_interscn_count['horseobs']    == 0  and \
                           obs_interscn_count['flyobs']      >= 1  and \
                           obs_interscn_count['commonobs']   == 0:

                           run.horsefly_input_graph.add_edge(p['idx'], q['idx'], edgetype='horseedge', weight=np.linalg.norm(qcd-pcd) )

                           # modify the node attributes at both ends
                           run.horsefly_input_graph.nodes[p['idx']]['point_type'] = p['point_type']
                           run.horsefly_input_graph.nodes[p['idx']]['coordinates'] = p['coordinates']
                          
                           run.horsefly_input_graph.nodes[q['idx']]['point_type'] = q['point_type']
                           run.horsefly_input_graph.nodes[q['idx']]['coordinates'] = q['coordinates']


                           ax.plot( [pcd[0],qcd[0]], [pcd[1],qcd[1]], 'r-', alpha = 0.2 ) 
          
      
      # Definition of method \verb|renderHorseflyInputGraph|
         
      def renderHorseflyInputGraph(self,fig,ax):
          pass
      

      def clearGraph(self,fig,ax):
          self.horsefly_input_graph = nx.Graph() # Input Graph is set to the empty null graph
          # TODO, clear the canvas completely and then redraw the current sites and obstacles stored
          # so that the drawn graph is cleared from the canvas, and the input is restored to its 
          # original prsitine state, so that other algorithms can be tried out. 
          pass

      def getTour(self, fig, ax):
            import copy

            str_phi = raw_input("What speed ratio do you want?: ")
            phi = float(str_phi)
            #--------------------------------------------------
            # Get graph which horse is allowed to travel along
            #-------------------------------------------------
            G_horse = copy.deepcopy(self.horsefly_input_graph)
            
            # Get list of flyedges
            flyedges = []
            for (u,v,etype) in G_horse.edges.data('edgetype'):
                  if etype == 'flyedge':
                        flyedges.append( (u,v) )

            # Remove flyedges from the graph
            for (u,v) in flyedges:
                  G_horse.remove_edge(u,v)

            # Remove isolated nodes
            G_horse.remove_nodes_from(list(nx.isolates(G_horse))) 
   
            #------------------------------------------------
            # Get graph which fly is allowed to travel along
            #------------------------------------------------
            G_fly   = copy.deepcopy(self.horsefly_input_graph) 

            # Get list of horseedges
            horseedges = []
            for (u,v,etype) in G_fly.edges.data('edgetype'):
                  if etype == 'horseedge':
                        horseedges.append( (u,v) )

            # Remove horseedges from the graph
            for (u,v) in horseedges:
                  G_fly.remove_edge(u,v)
            
            # Remove isolated nodes
            G_fly.remove_nodes_from(list(nx.isolates(G_fly))) 

            #---------------------------------------------------------------
            # Get indexes of nodes common to both the horse and fly
            # This is the list of potential rendezvous points for both the 
            # horse and fly
            #---------------------------------------------------------------
            horse_nodes_idxs = G_horse.nodes
            fly_nodes_idxs   = G_fly.nodes
            common_node_idxs = list(set(horse_nodes_idxs) & set(fly_nodes_idxs))

            #---------------
            # Get site info
            #---------------
            site_idxs        = [ node[0] for node in self.horsefly_input_graph.nodes(data=True) 
                                  if node[1]['point_type'] == 'site']
            
            site_coordinates = [np.asarray(self.horsefly_input_graph.nodes[node[0]]['coordinates']) 
                                for node in self.horsefly_input_graph.nodes(data=True) 
                                    if node[1]['point_type'] == 'site']

            numsites         = len(site_idxs)

            for node in self.horsefly_input_graph.nodes(data=True):
                  if node[1]['point_type'] == 'inithorseposn':
                        inithorseposn_idx = node[0]
                        inithorseposn     = np.asarray(self.horsefly_input_graph.nodes[node[0]]['coordinates'])
                        break

            #--------------------------------------------------------------
            # Run all pair shortest paths on both graphs to begin queries
            #--------------------------------------------------------------
            print Fore.CYAN, "Beginning all pairs shortest path computation.................."
            all_pairs_G_horse_paths      = dict(nx.shortest_path(G_horse, weight='weight'))
            all_pairs_G_horse_sp_lengths = dict(nx.shortest_path_length(G_horse, weight='weight'))
            
            all_pairs_G_fly_paths        = dict(nx.shortest_path(G_fly,weight='weight'))
            all_pairs_G_fly_sp_lengths   = dict(nx.shortest_path_length(G_fly, weight='weight'))
            print Fore.CYAN, "Finished  all pairs shortest path computation!", Style.RESET_ALL

            #-------------------------------------------------------------------------------------------------------------
            # Note that 
            # (1) A horse-path consists paths between successive rendezvous points. It is a list of lists
            # (2) A fly path is a list of tuples of size two, where each tuple consits of two lists, the 
            #     first one from rendezvous point to site, and second one from site back to some other rendezvous point
            # Thus both the horse-path and drone-path have the same length. Helpful during ploting
            #-------------------------------------------------------------------------------------------------------------
            horse_path_idxs, fly_path_idxs = [], []

            current_rendezvous_idx = inithorseposn_idx
            all_rendezvous_idxs    = [current_rendezvous_idx]
            
            unvisited_sites_idx   = copy.deepcopy(site_idxs)
            visited_sites_idx     = []
            visited_sites_coods   = [] 

            while unvisited_sites_idx:

               # Index of next site to rendezvous with   
               sidx_min = 0
               scood_min = None

               # Corresponding rendezvous point for that site
               gcn_imin = 0
               gcn_tmin = np.inf

               for site_idx, site_coords in zip(site_idxs, site_coordinates):

                  if site_idx in visited_sites_idx:  
                        continue
                  else:
                        #--------------------------------------------------------------------
                        # Find best rendezvous point by iterating over all the common nodes
                        #--------------------------------------------------------------------
                        cn_imin = 0
                        cn_tmin = np.inf
                        
                        for cn_idx in common_node_idxs:

                              #-----------------------------------------------------------------------
                              # Get distance of horse from current rendezvous node to the common node
                              #-----------------------------------------------------------------------
                              htime = all_pairs_G_horse_sp_lengths[current_rendezvous_idx][cn_idx]

                              ftime_to_site   = all_pairs_G_fly_sp_lengths[current_rendezvous_idx][site_idx]/phi
                              ftime_from_site = all_pairs_G_fly_sp_lengths[site_idx][cn_idx]/phi
                       
                              meeting_time_cn_idx = max(ftime_to_site+ftime_from_site, htime)

                              if meeting_time_cn_idx < cn_tmin:
                                    cn_imin = cn_idx
                                    cn_tmin = meeting_time_cn_idx

                        if cn_tmin < gcn_tmin:
                              gcn_tmin = cn_tmin
                              gcn_imin = cn_imin
                              sidx_min = site_idx
                              scood_min = site_coords
                              
               #-------------------------------------------------------
               # Now that the best rendezvous node has been detected
               # use the appropriate paths for the horse and the fly
               # to the rendezvous node and update the paths 
               #-------------------------------------------------------
               horse_path_idxs.append( all_pairs_G_horse_paths[current_rendezvous_idx][gcn_imin] )
                  
               fly_path_idxs.append( ( all_pairs_G_fly_paths[current_rendezvous_idx][sidx_min] ,\
                                      all_pairs_G_fly_paths[sidx_min][gcn_imin])        )

               #----------------------------------------------------------------------
               # Update the value of the current_rendezvous id for the next iteration
               #----------------------------------------------------------------------
               current_rendezvous_idx = gcn_imin
               all_rendezvous_idxs.append(current_rendezvous_idx)

               assert(sidx_min in unvisited_sites_idx)
               unvisited_sites_idx.remove(sidx_min)
               visited_sites_idx.append(sidx_min)
               visited_sites_coods.append(scood_min)


            rendezvous_points_coordinates = [G_horse.node[k]['coordinates'] 
                                             for k in all_rendezvous_idxs]
            rxs = [x for (x,y) in rendezvous_points_coordinates]
            rys = [y for (x,y) in rendezvous_points_coordinates]
            ax.plot(rxs,rys,'ro', markersize=12.0)

            #-------------------------------------------------------------------
            # Time for plotting! Plot the both the horse and fly paths ontop 
            # of the graph using the. Just render the fly path first and then 
            # the horse path. The horse path must be thicker and red, the 
            # fly path thinner and green. 
            #-------------------------------------------------------------------

            # Render fly path first. Just flatten the fly_path into a large list 
            # of nodes. And then render the edges in the graph using the coordinate
            # key-value pair stored at the dictionary. 

            # Init horseposn marker
            ax.add_artist(plt.Circle(inithorseposn, 0.03, color='k'))

            fly_path_idxs_flat   = [ idx for (p1,p2) in fly_path_idxs for idx in (p1+p2)]
            for idx1, idx2 in zip(fly_path_idxs_flat, fly_path_idxs_flat[1:]):

                  # find the coordinates of the points corresponding to the nodes 
                  # in the graph G_fly. 
                  p = G_fly.nodes[idx1]['coordinates']
                  q = G_fly.nodes[idx2]['coordinates']

                  # Render the edge
                  ax.plot([p[0],q[0]],[p[1],q[1]], linewidth=2.0, 
                                                   markersize=8, 
                                                   linestyle='dashed', 
                                                   color='blue')

            # Now render the horse path similar to the above
            horse_path_idxs_flat = [idx for p in horse_path_idxs for idx in p]
            for idx1, idx2 in zip(horse_path_idxs_flat, horse_path_idxs_flat[1:]):

                  # find the coordinates of the points corresponding to the nodes 
                  # in the graph G_horse. 
                  p = G_horse.nodes[idx1]['coordinates']
                  q = G_horse.nodes[idx2]['coordinates']

                  # Render the edge
                  ax.plot([p[0],q[0]],[p[1],q[1]], linewidth=4.0, 
                                                   markersize=10, 
                                                   linestyle='solid', 
                                                   color='red')

            assert(sorted(visited_sites_idx)==sorted(site_idxs))

            # Label the sites with the numbers 
            for k, (x,y) in zip(range(1,len(visited_sites_coods)+1), visited_sites_coods):
                  plt.annotate(str(k),(x,y),bbox={"boxstyle" : "circle", "color":"grey"},  fontsize=18)

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            ax.set_title("Number of sites: "+str(numsites) + "\nDrone Speed="+str(phi),
                         fontname='Times New Roman' ,fontsize= 20, ha='center')
            plt.show()



class Obstacle:
    def __init__(self, llcorner, width, height, figure=None, 
                 axes_object=None, obstype=None, diskcolor = 'crimson'):
        self.llcorner = llcorner
        self.width    = width
        self.height   = height
        self.fig      = figure
        self.ax       = axes_object
        self.obstype  = obstype

        # Only inserted disks will end up having this attribute
        if self.fig != None:
             self.canvas_patch =  mpl.patches.Rectangle( self.llcorner, self.width, 
                                                         self.height  , facecolor = diskcolor, 
                                                         alpha = 0.7 )

    def mplPatch(self, diskcolor= 'crimson' ):
        return  self.canvas_patch

    def getVertices(self):
         [x,y] = self.llcorner
         p_ll = [x            , y       ]
         p_lr = [x+self.width , y       ]
         p_ur = [x+self.width , y+self.height]
         p_ul = [x            , y+self.height]
         return [p_ll,p_lr,p_ur,p_ul]

    def intersectionWithSegment(self,p,q):
    
        [llv, _, urv, _] = self.getVertices()
        llv = Point_2(llv[0],llv[1])
        urv = Point_2(urv[0],urv[1])
        rect = Iso_rectangle_2(llv,urv)

        p = Point_2(p[0],p[1])
        q = Point_2(q[0],q[1])
        seg = Segment_2(p,q)

        interscn_object = intersection(rect,seg)
        interscn_p     = do_intersect(rect,seg)

        return (interscn_p, interscn_object)

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
             
             elif event.key in ['h' , 'H']: # `h` for horse
                  run.horse_obstacle_input_mode_p    = not (run.horse_obstacle_input_mode_p)
                  run.fly_obstacle_input_mode_p      = False
                  run.common_obstacle_input_mode_p   = False

    
             elif event.key in ['f' , 'F']: # `f` for fly
                  run.horse_obstacle_input_mode_p   = False
                  run.fly_obstacle_input_mode_p     = not (run.fly_obstacle_input_mode_p)
                  run.common_obstacle_input_mode_p  = False


             elif event.key in ['g' , 'G']: # 'g' lies between the f and h keys on the keybard signifying intersection
                  run.horse_obstacle_input_mode_p   = False
                  run.fly_obstacle_input_mode_p     = False
                  run.common_obstacle_input_mode_p  = not (run.common_obstacle_input_mode_p)
    
             elif event.key in ['d','D']: # `d` for discretize domain, using the obstacles we sprinkle 
                                          # some points and discretize everything
                  #background_grid_pts = [[np.random.rand(), np.random.rand()] for i in range(100)]
                  unit_interval_division = np.linspace(0.0,1.0,num=6) 
                  background_grid_pts = [ [x,y] for x in unit_interval_division for y in unit_interval_division  ]
                  run.makeHorseflyInputGraph(fig, ax, background_grid_pts)
                  plt.show()
                  #print list(run.horsefly_input_graph.nodes(data=True))
                  #utils_algo.print_list(list(run.horsefly_input_graph.edges(data=True)))


             elif event.key in ['a','A']: # run the greedy algorithm  to get a tour
                   run.getTour(fig,ax)


      return _keyPressHandler

# Implementation of \verb|wrapperEnterPoints|
   
def wrapperEnterPoints(fig,ax,run):
    def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           run.horse_obstacle_input_mode_p  == False        and \
           run.fly_obstacle_input_mode_p    == False        and \
           run.common_obstacle_input_mode_p == False        and \
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
                 

             elif event.button == 3:  
                 # Insert big red circle representing initial position of horse and fly
                 newinithorseposn  = (event.xdata, event.ydata)
                 run.inithorseposn = newinithorseposn  
                 patchSize         = (utils_graphics.xlim[1]-utils_graphics.xlim[0])/100.0

                 ax.add_patch( mpl.patches.Circle( newinithorseposn,radius = patchSize,
                                                   facecolor= '#D13131', edgecolor='black' ))
                 
                 print Fore.RED, "Initial positions of truck\n", 
                 print run.inithorseposn
                 print Style.RESET_ALL

             # Clear polygon patches and set up last minute \verb|ax| tweaks
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPoints

# Implementation of \verb|wrapperHoverObstacle|
   
def wrapperHoverObstacle(fig,ax, run):
        """ Wrapper for the call-back function _hoverObstacle
        """
        def _hoverObstacle(event, previous_patch = []):
          """ Bind the motion of the mouse with the movement of a disk to be placed.
          """
          if previous_patch != []:
              previous_patch.pop().remove() # This physically removes the patch from the screen and from memory

          if event.xdata != None and event.ydata!=None and (run.horse_obstacle_input_mode_p  or 
                                                            run.fly_obstacle_input_mode_p    or 
                                                            run.common_obstacle_input_mode_p):
              if run.horse_obstacle_input_mode_p: 
                    fcol = 'crimson'
    
              elif run.fly_obstacle_input_mode_p:
                     fcol = 'blue'

              elif run.common_obstacle_input_mode_p:
                     fcol = 'green'
    
              current_patch = mpl.patches.Rectangle((event.xdata,event.ydata),             \
                                                 width     = run.hovering_obstacle_width,  \
                                                 height    = run.hovering_obstacle_height,  \
                                                 facecolor = fcol,   \
                                                 alpha     = 0.5)
              previous_patch.append(current_patch)
              ax.add_patch(current_patch)

              fig.canvas.draw()
        return _hoverObstacle

# Implementation of \verb|wrapperPlaceObstacle|
   
def wrapperPlaceObstacle(fig,ax,run):
        def _placeObstacle(event):
            """ Double-clicking Button 1 inserts the hovering disk
                into the current arrangement and onto the canvas
            """
            if event.name     == 'button_press_event' and \
               event.button   == 1                    and \
               event.dblclick == True                 and \
               event.xdata    != None                 and \
               event.ydata    != None                 and \
               (run.horse_obstacle_input_mode_p or run.fly_obstacle_input_mode_p or run.common_obstacle_input_mode_p) :

                if run.horse_obstacle_input_mode_p: 
                    fcol = 'crimson'
                    obstype = 'horseobs'
    
                elif run.fly_obstacle_input_mode_p:
                     fcol = 'blue'
                     obstype = 'flyobs'

                elif run.common_obstacle_input_mode_p:
                     fcol = 'green'
                     obstype = 'commonobs' 
 
                # Update the current disk list
                run.obstacle_list.append( Obstacle( llcorner = [event.xdata,event.ydata], \
                                             width = run.hovering_obstacle_width, \
                                             height = run.hovering_obstacle_height, \
                                             figure      = fig, \
                                             axes_object = ax,
                                             obstype     = obstype  ,
                                             diskcolor   = fcol)  )

                # Add representation of the disk appended to the canvas and show the updated count
                ax.add_patch( run.obstacle_list[-1].mplPatch(fcol))

                # Render the canvas
                fig.canvas.draw()
        return _placeObstacle
   
# Implementation of \verb|wrapperResizeHoveringObstacle|
   
def wrapperResizeHoveringObstacle(fig,ax,run):
        """ Wrapper for the call-back function _resizeHoveringObstacle
        """
        def _resizeHoveringObstacle(event):
            """ Each key-press increments or decrements by a fixed amount
            the radius of the hovering disk. Change the frozenset global config
            GC dictionary for changing the increment and decrement deltas 
            corresponding to each key-press
            """

            # This used to be in the arguments to _resizeHoveringObstacle
            previous_patch=[]

            # Increase hovering disk radius
            if  event.key  == "shift":
                run.hovering_obstacle_height += 0.05
                run.hovering_obstacle_width  += 0.05

                current_patch = mpl.patches.Rectangle((event.xdata,event.ydata),         \
                                                   width     = run.hovering_obstacle_width,  \
                                                   height    = run.hovering_obstacle_height, \
                                                   facecolor = 'black', \
                                                   alpha=0.2)
                previous_patch.append(current_patch)
                ax.add_patch(current_patch)

                fig.canvas.draw()

            elif event.key == "control" and \
                run.hovering_obstacle_width >= 2.0 * 0.05:

                run.hovering_obstacle_width  -= 0.05
                run.hovering_obstacle_height -= 0.05

                current_patch = mpl.patches.Rectangle((event.xdata,event.ydata),           \
                                                   width     = run.hovering_obstacle_width,  \
                                                   height    = run.hovering_obstacle_height, \
                                                   facecolor = 'black', \
                                                   alpha=0.2)
                previous_patch.append(current_patch)
                ax.add_patch(current_patch)

                fig.canvas.draw()

            while len(previous_patch) != 0:
                previous_patch.pop().remove()

        return _resizeHoveringObstacle

   

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
        
    hoverObstacle  = wrapperHoverObstacle(fig,ax, run)
    fig.canvas.mpl_connect('motion_notify_event', hoverObstacle )
        
    placeObstacle  = wrapperPlaceObstacle(fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , placeObstacle )

    resizeHoveringObstacle  = wrapperResizeHoveringObstacle(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', resizeHoveringObstacle)

    plt.show()
    
