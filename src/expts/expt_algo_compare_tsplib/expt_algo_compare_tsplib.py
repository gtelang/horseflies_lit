#/usr/bin/python
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import sys, os, time
import yaml
from colorama import Fore, Style
import io

sys.path.append('../../lib')
import problem_classic_horsefly as chf
import utils_algo

def normalize_to_unit_square( points ):
    """ Given a list of points, map it to the 
    unit square [0,1] x [0,1] isometrically
    This is because all my algorithms by default
    work on points in the unit square. 
    """
    xs = [float(x) for (x,_) in points]
    ys = [float(y) for (_,y) in points]

    xleft, xright  = min(xs), max(xs)
    ylower, yupper = min(ys), max(ys)

    # Shift all the points so that left most point 
    # is on the y-axis and lower-most point on the x-axis
    xs = [x-xleft  for x in xs]
    ys = [y-ylower for y in ys]
    
    scaling_factor = max( xright-xleft, yupper-ylower )

    xs = [x/scaling_factor  for x in xs]
    ys = [y/scaling_factor  for y in ys]

    return zip(xs, ys)

# List of YAML files corresponding to Euclidean instances.
filenames     = os.listdir('./euclidean_instances_yaml')

# Strip each such filename of the .yml extension
instancenames = [ os.path.splitext(f)[0] for f in filenames ] 

phi           = 3.0        # Arbitrarily chosen
inithorseposn = (0.5, 0.5) # Middle of the square
statistics    = {}         # Each key-value pait is recorded to a separate yaml file; 
                           # the keys of the dictionary are the instance names

original_tsp_lib_instance_filenames = os.listdir('./tsplib_symmetric_tsp_instances')

# Create a dedicated directory to store results
import errno
import subprocess

resultsdir = 'results'
try:
    os.mkdir(resultsdir)
except OSError as e:
    if e.errno == errno.EEXIST:
        subprocess.call(["rm", "-rf", " results"])
    else:
        raise

for iname in instancenames:
   if iname+'.opt.tour' in original_tsp_lib_instance_filenames:

      # create a dedicated directory to store results
      fname =  os.path.join('euclidean_instances_yaml',iname+'.yml')
      with open(fname, 'r') as stream:
         data = yaml.load(stream)
         coords = data['points']

         print "Processing ", Fore.CYAN, fname, Style.RESET_ALL

         # Normalize to unit square
         coords = normalize_to_unit_square( coords )
         statistics[iname] = {}

         # algo_list = ['algo_greedy', 
         #              'algo_greedy_incremental_insertion', 
         #              'algo_tsp_ordering', 
         #              'algo_kmeans', ]

         algo_list = ['algo_greedy']

         for algo_name in algo_list:
             
             if algo_name == 'algo_greedy':
                 algo = chf.algo_greedy
                 
             elif algo_name == 'algo_greedy_incremental_insertion':
                 algo = chf.algo_greedy_incremental_insertion
                 
             elif algo_name == 'algo_tsp_ordering': # use algo exact given specific ordering
                 algo = chf.algo_tsp_ordering

             # Run the algorithm and record tour-length 
             answer = algo(coords, inithorseposn, phi, 
                  post_optimizer=chf.algo_approximate_L1_given_specific_ordering,
                  write_algo_states_to_disk_p = False,
                  animate_schedule_p          = False)
          
             statistics[iname][algo_name] = answer

         utils_algo.write_to_yaml_file(statistics[iname], dir_name='results', file_name=iname+'.yml')
