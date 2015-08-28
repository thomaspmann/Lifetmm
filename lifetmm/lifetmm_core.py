# Import functions to allow backwards compatability
from __future__ import division, print_function, absolute_import
# TODO think I need to import range for pythons 2.7

from numpy import cos, inf, zeros, exp, conj, nan, isnan

import scipy as sp
import numpy as np

import sys
EPSILON = sys.float_info.epsilon # typical floating-point calculation error

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def I_mat(n1, n2, pol='u', th_0=0):
    # transfer matrix at an interface
    # TODO allow for different polarizations and angles
    if th_0 == 0: # no difference between polarizations for normal incidence
        r = (n1 - n2) / (n1 + n2)
        t = (2 * n1) / (n1 + n2)

    # elif pol == 'u':
    # elif pol == 'p':
    # elif pol == 's':

    ret = np.array([[1, r], [r, 1]], dtype=complex)
    ret /= t
    return ret


def L_mat(n, d, l):
    # propagation matrix
    # n = complex dielectric constant
    # d = thickness
    # l = wavelength
    xi = (2 * np.pi * d * n) / l
    L = np.array([[np.exp(complex(0, -1.0 * xi)), 0], [0, np.exp(complex(0, xi))]])
    return L


def TransferMatrix(d_list, n_list, lam_vac, th_0, pol, x_step=1):
    """
    :param pol: polarisation of incoming light. Either "s", "p" or "u".
    :param n_list: list of refractive indices (can be complex)
    :param d_list: list of thicknesses
    :param th_0: angle of incidence (0 for normal, pi/2 for glancing)
    :param lam_vac: vacuum wavelength of light
    :return: many things
    """

    # convert lists to numpy arrays if they're not already.
    n_list = np.array(n_list)
    d_list = np.array(d_list, dtype=float)

    # input tests
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
          or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    if type(x_step) != int:
        raise ValueError('x_step must be an integer otherwise. Reduce SI unit'
                         'inputs for thicknesses and wavelengths for greater resolution ')
    # TODO what does this do/are these necessary?
    # if (d_list[0] != inf) or (d_list[-1] != inf):
    #     raise ValueError('d_list must start and end with inf!')
    # if abs((n_list[0]*np.sin(th_0)).imag) > 100*EPSILON:
    #     raise ValueError('Error in n0 or th0!')

    num_layers = d_list.size

    # calculate incoherent power transmission through thick superstrate to coherent layers
    # TODO change for polarized light?
    n = n_list
    T_glass = abs((4.0 * 1.0 * n[0]) / ((1 + n[0]) ** 2))
    R_glass = abs((1 - n[0]) / (1 + n[0])) ** 2

    # calculate transfer marices, and field at each wavelength and position
    d_list[0] = 0                                          # Thickness of layer light is originating from not important
    d_cumsum = np.cumsum(d_list)                              # Start position of each layer
    x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)    # x positions to evaluate E field at
    # get x_mat (specifies what layer number the corresponding point in x_pos is in):
    comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
    comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))
    x_mat = sum(comp1 > comp2, 0)  # TODO might need to get changed to better match python indices - check

    # calculate the transfer matrices for incoherent reflection/transmission at the first interface
    S = I_mat(n[0], n[1])
    for layer in range(1, num_layers - 1):
        mL = L_mat(n[layer], d_list[layer], lam_vac)
        mI = I_mat(n[layer], n[layer + 1])
        S = np.asarray(np.mat(S) * np.mat(mL) * np.mat(mI))
    R = abs(S[1, 0] / S[0, 0]) ** 2
    T = abs((2 / (1 + n[0]))) / np.sqrt(1 - R_glass * R)

    # calculate all other transfer matrices
    E = np.zeros(len(x_pos), dtype=complex)  # Initialise E field
    for layer in range(1, num_layers):
        xi = 2 * np.pi * n[layer] / lam_vac
        dj = d_list[layer]
        x_indices = np.nonzero(x_mat == layer)
        x = x_pos[x_indices] - d_cumsum[layer - 1]
        # Calculate S_Prime
        S_prime = I_mat(n[0], n[1])
        for layer in range(2, layer + 1):
            mL = L_mat(n[layer - 1], d_list[layer - 1], lam_vac)
            mI = I_mat(n[layer - 1], n[layer])
            S_prime = np.asarray(np.mat(S_prime) * np.mat(mL) * np.mat(mI))
        # Calculate S_dprime (double prime)
        S_dprime = np.eye(2)
        for layer in range(layer, num_layers - 1):
            mI = I_mat(n[layer], n[layer + 1])
            mL = L_mat(n[layer + 1], d_list[layer + 1], lam_vac)
            S_dprime = np.asarray(np.mat(S_dprime) * np.mat(mI) * np.mat(mL))
        # Normalized Electric Field Profile
        num = T * (S_dprime[0, 0] * np.exp(complex(0, -1.0) * xi * (dj - x)) + S_dprime[1, 0] * np.exp(
            complex(0, 1) * xi * (dj - x)))
        den = S_prime[0, 0] * S_dprime[0, 0] * np.exp(complex(0, -1.0) * xi * dj) + S_prime[0, 1] * S_dprime[
            1, 0] * np.exp(complex(0, 1) * xi * dj)
        E[x_indices] = num / den

    # |E|^2
    E_square = abs(E[:]) ** 2

    # overall Reflection from device with incoherent reflections at first interface
    Reflection = R_glass + T_glass ** 2 * R / (1 - R_glass * R)

    # Absorption coefficient in 1/cm
    absorption = np.zeros(num_layers)
    for layer in range(1, num_layers):
        absorption[layer] = (4 * np.pi * np.imag(n[layer])) / (lam_vac * 1.0e-7)

    return {'E_square': E_square, 'absorption': absorption, 'x_pos': x_pos, # output functions of position
            'R': R, 'T': T,  # output overall properties of structure
            'd_list': d_list, 'th_0': th_0, 'n_list': n_list, 'lam_vac': lam_vac, 'pol': pol, # input structure
            }


class LifetimeTmm:
    """
    Putting it all together for easy use.

    Input the structure of the device and material refractive indices and then begin the fun!
    """
    def __init__(self, d_list, n_list):
        """
        Initilise with the structure of the material to be simulated
        """
        self.d_list = d_list
        self.n_list = n_list

    def __call__(self, lam_vac, th_0, pol='u', x_step=1):
        """
        Call the simulation for the specific structure with the wavelength(s) to be simulated and (optionally)
        the angle of incidence, polarization and resolution in x
        """
        return TransferMatrix(self.d_list, self.n_list, lam_vac, th_0, pol, x_step)

    def varyAngle(self, th_list):
        pass

    def varyThickness(self, layer, d_range):
        pass

    def varyRefractive(self, layer, n_range):
        pass
