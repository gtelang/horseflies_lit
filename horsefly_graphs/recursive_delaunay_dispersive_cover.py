import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import scipy as sp
import sys, os, time
import itertools
from   colorama import Fore, Style
import utils_graphics, utils_algo
import networkx as nx
from CGAL.CGAL_Kernel import Point_2, Segment_2, Iso_rectangle_2
from CGAL.CGAL_Kernel import do_intersect, intersection
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull, convex_hull_plot_2d


