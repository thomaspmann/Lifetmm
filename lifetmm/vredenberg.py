from numpy import cos, sin, inf, zeros, exp, conj, nan, isnan, pi, arcsin
import numpy as np


def Mj2(j):
    return


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

    # Using Snell's law evaluate the critical angle
    theta_crit = arcsin(n_clad/n_list[m])

    return max(theta_crit, pi/2)


def sumAngles(j):
    result = 0
    for theta in np.linspace(0, thetaCritical(m, n_list)):
        First_term = (3/(2*n_a)) * eps0 * Mj2(j) * abs(U_j(z_a))**2 / 3
        # TODO: use snell's law to evaluate theta_1 and theta_s for theta_m?
        H_term = (H(theta_1) + H(theta_s)) / (H(theta_1)*cos(theta_1) + H(theta_s)*cos(theta_s))
        weighting = (n_m**2) * cos(theta) * sin(theta)
        result += First_term * H_term * weighting
        return result


def A_enhanced():
    """
    Evaluate A_rad/A_a
    :return:
    """
    result = 0

    # Sum over modes
    for j in range(1, 5):
        result = (1/2) * sumAngles(j)


m = 3                   # Layer
n_m = n_list[m]         # Film refractive index
n_a = 1.54              # Bulk refractive index
eps0 = 8.854187817E-12  # Vacuum permitivity (F.m^-1)