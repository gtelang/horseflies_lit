import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys, os, time
from scipy.optimize import minimize

t = 0.0 # Initial position of truck and drone
sx, sy = 1.0, 1.0 # Coordinates of site to deliver to

# We set up a constrained optimization problem
# and solve it using sequential least squares 
# programming
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, None), (0, None))
res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
               constraints=cons)


fig, ax = plt.subplots()
ax.set_aspect(1.0)


# Plot initial position of truck and drone
# Plot horizontal supporting line
# Plot to and from trip of the drone

# Sort sites. 
