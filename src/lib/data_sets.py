#!/usr/bin/python
#....................................................
# This file has functions to generate data-sets 
# for testing TSP and Horsefly algorithms. The 
# functions are mostly patterned on the data-sets 
# from Jon Bentley's paper on Experimental Heuristics 
# for the TSP, some very minor modifications might 
# have been made. Jon does not assume his data-set 
# to lie in the unit-squre, but I do, for the the 
# purposes of visualization with Python, and to make 
# things convenient for myself. This file is meant
# to be imported as a library, but a main fucntion
# has been latched on at the end for visually
# debugging the generated data-sets
# https://dl.acm.org/citation.cfm?id=320186
#....................................................
import matplotlib.pyplot as plt
from   matplotlib import rc
import numpy as np
import scipy as sp
import random

def unpack_points(pts):
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    return xs, ys

def uni(N):
    """ Uniform within the unit square [0,1]^2
    """
    xs = np.random.rand(N)
    ys = np.random.rand(N)
    return map(np.asarray, zip(xs,ys))

def annulus(N):
    """ Uniform on a circle (Zero width annulus), placed regularly
    """
    shift = np.asarray([0.5,0.5])
    xs = [ np.cos(2*i*np.pi/N) * 0.5  + shift for i in range(N)]
    ys = [ np.sin(2*i*np.pi/N) * 0.5  + shift for i in range(N)]
    return map(np.asarray, zip(xs,ys))

def ball(N):
    """ Uniform inside a circle centered at (0.5,0.5)
    """
    pts   = []
    while len(pts) < N:
        x = np.random.rand() 
        y = np.random.rand() 
        if (x-0.5)**2 + (y-0.5)**2 < 0.5**2:
            pts.append( np.asarray([x,y]))
    assert len(pts) == N , ""
    return pts

def normal(N):
    """ Each dimension independent from Normal(0.5,0.1)
    """
    mu    = 0.5
    sigma = 0.10
    pts   = []
    while len(pts) < N:
        x = np.random.normal(mu, sigma)
        y = np.random.normal(mu, sigma)
        if  x > 0 and x < 1 and y>0 and y < 1:
            pts.append( np.asarray([x,y]))
    assert len(pts) == N , ""
    return pts

def spokes(N):
    """ N/2 poitns at (U[0,1],1/2) and N/2 points 
    at (1/2, U[0,1])
    """
    H = int(np.floor(N/2))
    V = int(np.ceil (N/2)) 

    assert H + V == N ,""
    
    xs1 = np.random.rand(H)
    ys1 = [0.5 for _ in range(H)]
    pts1= map(np.asarray, zip(xs1, ys1))

    xs2 = [0.5 for _ in range(V)]
    ys2 = np.random.rand(V)
    pts2= map(np.asarray, zip(xs2, ys2))

    points = map(np.asarray, pts1+pts2)
    assert len(points) == N, ""
    return points


def clusunif(N):
    """ Choose 10 points at random from the unit square
    then put a uniform distribution at each
    """
    numclus = 10
    a = 0.05
    b = 1 - 0.05
    xcens   = a + (b-a) * np.random.rand(numclus)
    ycens   = a + (b-a) * np.random.rand(numclus)
    pts     = []
    
    while len(pts) < N:
        idxr = random.randint(0,numclus-1)
        xc   = xcens[idxr]
        yc   = ycens[idxr]
        
        limx = min( xc-0, 1-xc ) / 3
        limy = min( yc-0, 1-yc ) / 3

        xr   = xc + limx * np.random.rand()
        yr   = yc + limy * np.random.rand()

        pts.append(np.asarray([xr,yr]))
    return pts


def grid(N, scale=1.3):
    """ Choose N points from a square frid that contains about scale*N
    points. 
    """
    K       = int(np.sqrt(np.ceil(scale*N)))
    gridpts = []
    for x in np.linspace(0,1,K):
        for y in np.linspace(0,1,K):
            gridpts.append(np.asarray([x,y]))
            
    idxs = np.random.choice(range(len(gridpts)), N, replace=False)
    return [gridpts[idx] for idx in idxs]
    

def genpointset(numpts, cloudtype):
    
    if cloudtype == 'uni':
        return uni(numpts)

    elif cloudtype == 'annulus':
         return annulus (numpts)

    elif cloudtype == 'ball':
        return ball(numpts)

    elif cloudtype == 'clusnorm':
        return clusnorm(numpts)
    
    elif cloudtype == 'normal':
        return normal(numpts)
    
    elif cloudtype == 'spokes':
        return spokes(numpts)

    elif cloudtype == 'grid':
        return grid(numpts, scale=2.0)

    

if __name__ == '__main__':
    
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    points = genpointset(300, cloudtype='grid')
    xs, ys = unpack_points(points)
    ax.scatter(xs,ys)
    plt.show()
