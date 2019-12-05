import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
import networkx as nx


def get_shortcutted_euler_tour(G, source):
  """ Given an Eulerian multigraph, get a shortcutted Euler Tour. 
  """
  def flatten_tuple_list(xs):
      flxs = []
      for elt in xs:
        flxs.append(elt[0])
        flxs.append(elt[1])
      return flxs
 
  def shortcut_list(xs):
      bag = []
      for elt in xs:
          if elt not in bag:
              bag.append(elt)
      return bag

  assert G.is_multigraph(), "this should be a multigraph"
  tour     = list(nx.eulerian_circuit(G, source=source))
  shortcut = shortcut_list(flatten_tuple_list(tour))
  #print "Euler Tour ", tour
  #print "Shortcutted sequence ", shortcut
  return shortcut


def double_edges_of_graph(G):
    """ Pass an undirected non multigraph graph and double all the edges returning a multigraph. 
    """
    assert not(G.is_multigraph()), "graph passed must simple undirected"
    G     = nx.MultiGraph(G)
    edges =  G.edges(data=True)
    for e in edges:
        G.add_edge(e[0], e[1]) # what about also adding the data associated with the doubled edge? 
    return G

# # Construct a graph, double the edges 
G = nx.MultiGraph()
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,5)
G.add_edge(5,2)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,5)
G.add_edge(5,2)
G.add_edge(5,6)
G.add_edge(5,6)

get_shortcutted_euler_tour( double_edges_of_graph(G), source=2)
nx.draw_networkx(G)
plt.show()

