# Import functions to allow backwards compatability
from __future__ import division, print_function, absolute_import
# TODO think I need to import range for pythons 2.7

from numpy import cos, inf, zeros, exp, conj, nan, isnan

import scipy as sp
import numpy as np

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def q_j(nj, n0, th_0):
    """
    Calculate the q factor for layer j
    :param n0:
    :param nj:
    :param th_0: Angle of incidence in radians
    :return:
    """

    return np.sqrt(nj**2 - n0**2 * np.sin(th_0)**2)


def I_mat(n1, n2, n0, pol, th_0):
    """
    Calculates the interference matrix between two layers.
    :param n1: First layer refractive index
    :param n2: Second layer refractive index
    :param n0: Refractive index of incident transparent medium
    :param pol: Polarisation of incoming light ('s' or 'p')
    :param th_0: Angle of incidence of light (0 for normal, pi/2 for glancing)
    :return: I-matrix
    """

    if th_0 == 0:  # no difference between polarizations for normal incidence
        r = (n1 - n2) / (n1 + n2)
        t = (2 * n1) / (n1 + n2)

    else:
        # transfer matrix at an interface
        q1 = q_j(n1, n0, th_0)
        q2 = q_j(n2, n0, th_0)

        if pol == 's':
            r = (q1 - q2) / (q1 + q2)
            t = (2 * q1) / (q1 + q2)

        elif pol == 'p':
            r = (q1 * n2**2 - q2 * n1**2) / (q1 * n2**2 + q2 * n1**2)
            t = (2 * n1 * n2 * q1) / (q1 * n2**2 + q2 * n1**2)

        else:
            raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                             " not 90$\degree$s")

    if t == 0:
        raise ValueError('Transmission is zero. Check input parameters.')

    return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)


def L_mat(n, d, lam_vac, n0, th_0):
    """
    Calculates the propagation.
    :param n: complex dielectric constant
    :param d: thickness
    :param lam_vac: wavelength
    :return:  L-matrix
    """

    eps = (2 * np.pi * q_j(n, n0, th_0)) / lam_vac

    return np.array([[np.exp(complex(0, -1.0 * eps * d)), 0],
                     [0, np.exp(complex(0, eps * d))]])


def TransferMatrix(d_list, n_list, lam_vac, th_0, pol, x_step=1, reverse=False, glass=False):
    """
    Evaluate the transfer matrix over the entire structure.
    :param pol: polarisation of incoming light ('u', 's' or 'p')
    :param n_list: list of refractive indices for each layer (can be complex)
    :param d_list: list of thicknesses for each layer
    :param th_0: angle of incidence (0 for normal, pi/2 for glancing)
    :param lam_vac: vacuum wavelength of light
    :param glass: Bool. If there is a thick superstrate present then make true as interference
                        in this layer is negligible
    :return: Dictionary of all input and output params related to structure
    """
    # convert lists to numpy arrays if they're not already.
    n_list = np.array(n_list, dtype=complex)
    d_list = np.array(d_list, dtype=float)

    # input tests
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1) or (hasattr(th_0, 'size')
                                                            and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if type(x_step) != int:
        raise ValueError('x_step must be an integer otherwise. Reduce SI unit'
                         'inputs for thicknesses and wavelengths for greater resolution ')
    if th_0 >= 90 or th_0 <= -90:
        raise ValueError('The light is not incident on the structure. Check input theta '
                         '(0 <= theta < 90')

    # Flip structure if the optional argument 'reverse' is True
    if reverse:
        d_list = d_list[::-1]
        n_list = n_list[::-1]

    # Set first and last layer thicknesses to zero (from inf) - helps for summing later in program
    d_list[0] = 0
    d_list[-1] = 0

    num_layers = d_list.size
    n = n_list

    d_cumsum = np.cumsum(d_list)                              # Start position of each layer
    x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)    # x positions to evaluate E field at

    # get x_mat (specifies what layer number the corresponding point in x_pos is in):
    comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
    comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))

    x_mat = sum(comp1 > comp2, 0)  # TODO might need to get changed to better match python indices - check

    # calculate the total system transfer matrix S
    S = I_mat(n[0], n[1], n[0], pol, th_0)
    for layer in range(1, num_layers - 1):
        mL = L_mat(n[layer], d_list[layer], lam_vac, n[0], th_0)
        mI = I_mat(n[layer], n[layer + 1], n[0], pol, th_0)
        S = np.asarray(np.mat(S) * np.mat(mL) * np.mat(mI))

    # JAP Vol 86 p.487 Eq 9 and 10: Power Reflection and Transmission
    # R = abs(S[1, 0] / S[0, 0]) ** 2
    # T = abs(1 / S[0, 0]) ** 2  # note this is incorrect https://en.wikipedia.org/wiki/Fresnel_equations
    T = R = 1

    # Calculate incoherent power transmission into the glass - if glass is the ambient medium input
    # See Griffiths "Intro to Electrodynamics 3rd Ed. Eq. 9.86 & 9.87
    # If first layer in n_list is air (n=1) then these don't do anything
    if glass and not reverse:
        T_glass = abs((4.0 * 1.0 * n[0]) / ((1 + n[0]) ** 2))
        R_glass = abs((1 - n[0]) / (1 + n[0])) ** 2

        # Transmission of field through glass superstrate to multilayered stack)
        # See Griffiths 9.85 + multiple reflection geometric series
        T = abs((2 / (1 + n[0]))) / np.sqrt(1 - R_glass * R)
        # overall Reflection from device with incoherent reflections at first interface
        R = R_glass + T_glass ** 2 * R / (1 - R_glass * R)

    # calculate primed transfer matrices for info on field inside the structure
    E = np.zeros(len(x_pos), dtype=complex)  # Initialise E field
    E_avg = np.zeros(num_layers)
    for layer in range(1, num_layers):
        eps = (2 * np.pi * q_j(n[layer], n[0], th_0)) / lam_vac
        dj = d_list[layer]
        x_indices = np.nonzero(x_mat == layer)
        x = x_pos[x_indices] - d_cumsum[layer - 1]
        # Calculate S_Prime
        S_prime = I_mat(n[0], n[1], n[0], pol, th_0)
        for layerind in range(2, layer + 1):
            mL = L_mat(n[layerind - 1], d_list[layerind - 1], lam_vac, n[0], th_0)
            mI = I_mat(n[layerind - 1], n[layerind], n[0], pol, th_0)
            S_prime = np.asarray(np.mat(S_prime) * np.mat(mL) * np.mat(mI))

        # Calculate S_dprime (double prime)
        S_dprime = np.eye(2)
        for layerind in range(layer, num_layers - 1):
            mI = I_mat(n[layerind], n[layerind + 1], n[0], pol, th_0)
            mL = L_mat(n[layerind + 1], d_list[layerind + 1], lam_vac, n[0], th_0)
            S_dprime = np.asarray(np.mat(S_dprime) * np.mat(mI) * np.mat(mL))

        # Electric Field Profile
        num = S_dprime[0, 0] * np.exp(complex(0, -1.0) * eps * (dj - x)) + S_dprime[1, 0] * np.exp(
            complex(0, 1) * eps * (dj - x))
        den = S_prime[0, 0] * S_dprime[0, 0] * np.exp(complex(0, -1.0) * eps * dj) + S_prime[0, 1] * S_dprime[
            1, 0] * np.exp(complex(0, 1) * eps * dj)
        E[x_indices] = num / den

        # TODO check is using T only for glass superstrate correct here?
        # T used is just a normalizing factor which shouldn't
        if glass:
            E[x_indices] *= T

        # Calculate the average E field in the layers between the ambient and substrate medium
        if not d_list[layer] == 0:
            E_avg[layer] = sum(abs(E[x_indices])**2) / (x_step*d_list[layer])  # Average E field inside the layer

    # |E|^2
    E_square = abs(E[:]) ** 2

    # Absorption coefficient in 1/cm
    absorption = np.zeros(num_layers)
    for layer in range(1, num_layers):
        absorption[layer] = (4 * np.pi * np.imag(n[layer])) / (lam_vac * 1.0e-7)

    # Flip back to normal
    if reverse:
        d_list = d_list[::-1]
        n_list = n_list[::-1]
        E_square = E_square[::-1]
        E_avg = E_avg[::-1]
        E = E[::-1]
        absorption = absorption[::-1]

    return {'E_square': E_square, 'absorption': absorption, 'x_pos': x_pos,  # output functions of position
            'R': R, 'T': T, 'E': E, 'E_avg': E_avg,  # output overall properties of structure
            'd_list': d_list, 'th_0': th_0, 'n_list': n_list, 'lam_vac': lam_vac, 'pol': pol,  # input structure
            }


class LifetimeTmm:
    """
    Putting it all together for easy use.
    Input the structure of the device and material refractive indices and then begin the fun!
    """
    def __init__(self, d_list, n_list, x_step=1):
        """
        Initilise with the structure of the material to be simulated
        """
        self.d_list = d_list
        self.n_list = n_list

        # TODO problem if one of the thicknesses in d_list is inf
        self.x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)

    def __call__(self, lam_vac, th_0, pol, x_step=1):
        """
        Call the simulation for the specific structure with the wavelength(s) to be simulated and (optionally)
        the angle of incidence, polarization and resolution in x
        """
        return TransferMatrix(self.d_list, self.n_list, lam_vac, th_0, pol, x_step)

    def reverse(self, lam_vac, th_0, pol='u', x_step=1):
        """
        evaluate the transfer matrix with light incident on the last layer (i.e. flip structure)
        """
        d_list_rev = self.d_list[::-1]
        n_list_rev = self.n_list[::-1]

        return TransferMatrix(d_list_rev, n_list_rev, lam_vac, th_0, pol, x_step)

    def fluorescence(self):
        """
        Calculate for all wavelengths (weighted) and angles
        """
        pass

    def varyAngle(self, th_list):
        pass

    def varyThickness(self, layer, d_range):
        pass

    def varyRefractive(self, layer, n_range):
        pass
