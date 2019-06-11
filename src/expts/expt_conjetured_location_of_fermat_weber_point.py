import geometric_median as gm
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import os
from colorama import Fore, Style

fig = plt.figure()
ax = plt.subplot(111)
os.system("rm fermat_weber_plots/*")  


# Get the delaunay triangulation of the point-set
points = np.random.rand(7,2)
#points = np.asarray(map(lambda th : np.asarray([np.cos(th), np.sin(th)]), 
#                        2*np.pi*np.random.rand(10)))
print points

glob_fw_center = np.asarray(gm.geometric_median(points))

filenum = 0
while len(points) >= 4 and len(points) <= 100:

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
        fwc = gm.geometric_median([p,q,r])
        print Fore.YELLOW, "fwc=", fwc, Style.RESET_ALL
        fw_centers.append(fwc)

    fw_centers = np.asarray(fw_centers)
    print Fore.RED, "fw_centers", fw_centers, Style.RESET_ALL
    

    # Plot these centers
    ax.plot(fw_centers[:,0], fw_centers[:,1], 'ro')
    ax.plot(glob_fw_center[0], glob_fw_center[1], 'ks', markersize=5)

    # Get the convex hull of the centers
    hull = ConvexHull(fw_centers)

    for simplex in hull.simplices:
       plt.plot(fw_centers[simplex, 0], fw_centers[simplex, 1], 'r-',lw=1)

    plt.savefig('fermat_weber_plots/myplot_' + str(filenum).zfill(4) + '.png',
                bbox_inches='tight', dpi=100)
 
    filenum += 1

    # Current fermat-weber centers are points for next iteration
    points = fw_centers






