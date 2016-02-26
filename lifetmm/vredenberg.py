import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from .walpita import *

from numpy import cos, sin, inf, zeros, exp, conj, nan, isnan, pi

def prepareStruct(d_layers, n_layers, Lz = 1E5):
    """
    Insert pseudo layers of the ambient and substrate layers into the structure. Used for averaging.
    :param d_layers: list of thicknesses of each layer
    :param n_layers: list of refractive index for each layer
    :param Lz: Arbitrarily large number
    :return:
    """
    d1 = Lz/(2*n_layers[0])
    ds = Lz/(2*n_layers[-1])

    d_layers = np.insert(d_layers, 1, [d1])
    d_layers = np.insert(d_layers, -1, [ds])

    n_layers = np.insert(n_layers, 1, n_layers[0])
    n_layers = np.insert(n_layers, -1, n_layers[-1])
    return d_layers, n_layers


def snell(n_1, n_2, th_1):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """
    # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
    # give different results e.g. for arcsin(2).)
    # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
    # arcsin(2) due to branch cut
    return sp.arcsin(np.real_if_close(n_1*np.sin(th_1) / n_2))


def Mj2(j, theta_1, theta_s, theta):
    # eps0 = 8.854187817E-12  # Vacuum permitivity (F.m^-1)
    eps0 = 1  # Cancels out with bit before Mj2
    n_1 = n_list[0]
    n_s = n_list[-1]

    E_field = transfer_matrix(j, d_list, n_list, lam_vac, theta)
    E_1 = E_field[1]
    E_s = E_field[-2]

    denom = eps0 * (n_1*H(theta_1)*(abs(E_1)**2) + n_s*H(theta_s)*abs(E_s)**2)
    return 2 / denom


def H(theta):
    if np.isreal(theta):
        return 1
    else:
        return 0


def thetaCritical(m, n_list):
    """
    :param m: layer containing the emitting atom
    :param n_list: list of refractive indices of the layers
    :return: Return the angle at which TIRF occurs between the layer containing the atom and the cladding with
    the largest refractive index, or pi/2, whichever comes first.
    """

    # Evaluate largest refractive index of claddings
    n_clad = max(n_list[0], n_list[-1])

    # Using Snell's law evaluate the critical angle or return pi/2 if does not exist
    if n_clad/n_list[m] < 1:
        return np.arcsin(n_clad/n_list[m])
    else:
        return pi/2


def func(theta_m, d_list, n_list, m, j):
    n_m = n_list[m+1]                   # Film refractive index
    theta_1 = snell(n_m, n_list[0], theta_m)
    theta_s = snell(n_m, n_list[-1], theta_m)

    # Evaluate first bracketed term
    # eps0 = 8.854187817E-12              # Vacuum permittivity (F.m^-1)
    eps0 = 1  # Cancels out with Mj2
    n_a = 1.5                           # Bulk refractive index
    U_j = transfer_matrix(gamma, d_list, n_list, lam_vac, theta)

    first_term = (3/(2*n_a))*eps0*Mj2(j, theta_1, theta_s, theta_m)*(abs(U_j)**2 / 3)

    # Evaluate H bracketed term
    H_term = (H(theta_1) + H(theta_s)) / (H(theta_1)*cos(theta_1) + H(theta_s)*cos(theta_s))
    H_term = H_term.real

    weighting = (n_m**2)*cos(theta_m)*sin(theta_m)
    return first_term * H_term * weighting


def A_rad(d_list, n_list, m):
    theta_max = thetaCritical(m, n_list)
    enhancement = 0
    # Sum over modes and directions
    for j in range(1, 4):
        print('Evaluating mode = %s' % j)
        int_angles = integrate.quad(func, 0, theta_max, args=(d_list, n_list, m, j))
        enhancement += ((1/2)*int_angles)
    return enhancement


if __name__ == "__main__":

    # list of layer thicknesses in nm
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1, 1.5, 3]
    # Active layer
    m = 1

    # Prepare the structure for calculation
    d_list, n_list = prepareStruct(d_list, n_list)

    result = A_rad(d_list, n_list, m)

    plt.figure()

    # Plot layer lines
    dsum = np.cumsum(d_list[1:-2])
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)

    # #Plot E Field
    plt.plot(result)
    plt.show()
