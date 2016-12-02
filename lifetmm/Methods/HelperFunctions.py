import logging

import numpy as np
import scipy as sp

log = logging.getLogger(__name__)


#####################################################################
# Recursive root finding functions
# http://stackoverflow.com/questions/13054758/python-finding-multiple-roots-of-nonlinear-equation
def root_search(f, a, b, dx):
    """
    Find an interval f(x1) and f(x2=x1+dx) closest to a in the interval
    [a,b] that contains a root. Starting at x1=a and x2=a+dx keep incrementing
    x1 and x2 by dx until a sign change in f is observed or x1>=b is reached.
    """
    assert a <= b, ValueError('a must be less than b')
    x1 = a
    f1 = f(x1)
    x2 = a + dx
    f2 = f(x2)
    while f1 * f2 > 0.0:
        if x1 >= b:
            return None, None
        x1 = x2
        f1 = f2
        x2 = x2 + dx
        f2 = f(x2)
    return x1, x2


def roots(f, a, b, num=20000, verbose=True):
    """
    Find roots of f within the interval [a,b]. Interval is discretised
    into num equal elements, dx, and a root is searched for within each dx.
    """
    import math
    from scipy.optimize import brentq

    if verbose:
        print('The roots on the interval [{:f}, {:f}] are:'.format(a, b))
    else:
        logging.debug('The roots on the interval [{:f}, {:f}] are:'.format(a, b))

    results = []
    while 1:
        dx = abs(a - b) / num  # alternatively use dx=eps where eps is arg
        x1, x2 = root_search(f, a, b, dx)
        if x1 is not None:
            a = x2
            root = brentq(f, x1, x2, rtol=1e-5)
            if root != 0:
                # Root is only as accurate as the width of the
                # element as there could be multiple roots within each dx
                root = round(root, -int(math.log(dx, 10)))
                results.append(root)

                if verbose:
                    print('{:.4f}'.format(root))
                else:
                    logging.debug('{:.4f}'.format(root))
        else:
            if verbose:
                print('\nDone')
            return np.array(results)


#####################################################################
# Optical Functions
def snell(n_1, n_2, th_1):
    """
    Return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """
    # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
    # give different results e.g. for arcsin(2).)
    # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
    # arcsin(2) due to branch cut
    return sp.arcsin(np.real_if_close(n_1 * np.sin(th_1) / n_2))


def lambda2omega(lambda_):
    """
    Convert wavelength to omega
    """
    from scipy.constants import lambda2nu
    from math import pi
    return 2 * pi * lambda2nu(lambda_)


def omega2lambda(omega):
    """
    Convert omega to wavelength
    """
    from scipy.constants import nu2lambda
    from math import pi
    nu = omega / (2 * pi)
    return nu2lambda(nu)


#####################################################################
# 2x2 Matrix Determinant - can be complex unlike: from numpy.linalg import det
def det(matrix):
    return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]
