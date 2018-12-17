
import numpy as np
import random
from colorama import Fore
from colorama import Style


def vector_chain_from_point_list(pts):
    """ Given a list of points [p0,p1,p2,....p(n-1)]
    Make it into a list of numpy vectors 
    [p1-p0, p2-p1,...,p(n-1)-p(n-2)]
    
    Points should be lists or tuples of length 2
    """
    vec_chain = []
    for pair in zip(pts, pts[1:]):
        tail= np.array (pair[0])
        head= np.array (pair[1])
        vec_chain.append(head-tail)

    return vec_chain

def length_polygonal_chain(pts):
    """ Given a list of points [p0,p1,p2,....p(n-1)]
    calculate the length of its segments. 

    Points should be lists or tuples of length 2

    If no points or just one point is given in the list of
    points, then 0 is returned.
    """
    vec_chain = vector_chain_from_point_list(pts)

    acc = 0
    for vec in vec_chain:
        acc = acc + np.linalg.norm(vec)
    return acc

def pointify_vector (x):
    """ Convert a vector of even length 
    into a vector of points. i.e.
    [x0,x1,x2,...x2n] -> [[x0,x1],[x2,x3],,..[x2n-1,x2n]]
    """
    if len(x) % 2 == 0:
        pts = []
        for i in range(len(x))[::2]:
            pts.append( [x[i],x[i+1]] )
        return pts
    else :
        sys.exit('List of items does not have an even length to be able to be pointifyed')

def flatten_list_of_lists(l):
    """ Flatten vector
      e.g.  [[0,1],[2,3],[4,5]] -> [0,1,2,3,4,5]
    """
    return [item for sublist in l for item in sublist]


def print_list(xs):
    """ Print each item of a list on new line
    """
    for x in xs:
        print x

def partial_sums( xs ):
    """
    List of partial sums
    [4,2,3] -> [4,6,9]
    """
    psum = 0
    acc = []
    for x in xs:
        psum = psum+x
        acc.append( psum )

    return acc

def are_site_orderings_equal(sites1, sites2):
    """
    For two given lists of points test if they are 
    equal or not. We do this by checking the Linfinity
    norm.
    """
    
    for (x1,y1), (x2,y2) in zip(sites1, sites2): 
        if (x1-x2)**2 + (y1-y2)**2 > 1e-8:

            print Fore.BLUE+ "Site Orderings are not equal"
            print sites1
            print sites2
            print '-------------------------' + Style.RESET_ALL
            return False

    return True
        
    print "\n\n\n\n---------------------"


def bunch_of_random_points(numpts):
    cluster_size = int(np.sqrt(numpts)) 
    numcenters   = cluster_size
    
    import scipy
    import random
    centers = scipy.rand(numcenters,2).tolist()

    scale = 4.0
    points = []
    for c in centers:
        cx = c[0]
        cy = c[1]

        sq_size      = min(cx,1-cx,cy, 1-cy)
        cluster_size = int(np.sqrt(numpts)) 
        loc_pts_x    = np.random.uniform(low=cx-sq_size/scale, 
                                         high=cx+sq_size/scale, 
                                         size=(cluster_size,))
        loc_pts_y    = np.random.uniform(low=cy-sq_size/scale, 
                                         high=cy+sq_size/scale, 
                                         size=(cluster_size,))

        points.extend(zip(loc_pts_x, loc_pts_y))

    num_remaining_pts = numpts - cluster_size * numcenters

    remaining_pts = scipy.rand(num_remaining_pts, 2).tolist()
    points.extend(remaining_pts)
    
    return points

