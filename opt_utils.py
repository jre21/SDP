from __future__ import division, print_function
import cvxpy as cvx
import random
import numpy

def rand_matrix(m, n, sigma=None, positive=False, symmetric=False, integer=False):
    """Generate a matrix of random values.

    sigma: standard deviation.  Defaults to 1 for floats or 100 for ints.
    positive: if true, use only non-negative entries.
    symmetrix: if true, return a symmetric matrix.
    integer: if true, return a matrix of integers.
    """
    if integer:
        if sigma is None:
            sigma = 1000
        ret = numpy.empty((m,n), dtype=int)
        for i in range(m):
            if(symmetric):
                for j in range(i+1):
                    if positive:
                        ret[i,j] = ret[j,i] = round(abs(random.gauss(0,sigma)))
                    else:
                        ret[i,j] = ret[j,i] = round(random.gauss(0,sigma))
            else:
                for j in range(n):
                    if positive:
                        ret[i,j] = round(abs(random.gauss(0,sigma)))
                    else:
                        ret[i,j] = round(random.gauss(0,sigma))
    else:
        if sigma is None:
            sigma = 1
        ret = numpy.empty((m,n))
        for i in range(m):
            if(symmetric):
                for j in range(i+1):
                    if positive:
                        ret[i,j] = ret[j,i] = abs(random.gauss(0,sigma))
                    else:
                        ret[i,j] = ret[j,i] = random.gauss(0,sigma)
            else:
                for j in range(n):
                    if positive:
                        ret[i,j] = abs(random.gauss(0,sigma))
                    else:
                        ret[i,j] = random.gauss(0,sigma)
    return ret
