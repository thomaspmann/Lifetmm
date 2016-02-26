import scipy as sp
import numpy as np

from numpy import exp, inf, pi, cos, sin, tan, arctan


def make_2x2_array(a, b, c, d, dtype=complex):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=complex)", but ten times faster
    """
    my_array = np.empty((2, 2), dtype=dtype)
    my_array[0, 0] = a
    my_array[0, 1] = b
    my_array[1, 0] = c
    my_array[1, 1] = d
    return my_array


def snell(n_1,n_2,th_1):
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

def list_snell(n_list,th_0):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
    # give different results e.g. for arcsin(2).)
    # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
    # arcsin(2) due to branch cut
    return sp.arcsin(np.real_if_close(n_list[0]*np.sin(th_0) / n_list))


def p_z(j, gamma, theta, n_o_list, n_e_list):

    k = 2*pi / lam_vac  # Free-space wave vector
    n_o = n_o_list[j]
    n_e = n_e_list[j]

    # TE Mode
    if gamma == 0:
        return k*n_o*sin(theta)
    # TM Mode
    elif gamma == 1:
        theta_eff = arctan(tan(theta)*(n_o/n_e)**2)
        n_eff = (n_e*n_o)/((n_e*sin(theta_eff))**2 + (n_o*cos(theta_eff))**2)**0.5
        beta = k*n_eff*cos(theta_eff)
        return (beta**2 - (k*n_e)**2)**0.5
    else:
        raise ValueError("Gamma must be 0 or 1")


def z(j):
    return


def gammaMatrix(j, gamma, theta, n_o_list, n_e_list):
    n_jo = n_list[j]
    gam = p_z(j, gamma, theta, n_o_list, n_e_list)/n_jo**(gamma*2)
    return make_2x2_array(1, 1, -gam, gam)


def expMatrix(j, gamma, theta, n_o_list, n_e_list):
    x = p_z(j, gamma, theta, n_o_list, n_e_list)*(z(j)-z(j-2))
    return make_2x2_array(exp(-x), exp(x), exp(-x), exp(x))


def layerMatrix(j, gamma, theta, n_o_list, n_e_list):
    return expMatrix(j, gamma, theta, n_o_list, n_e_list) @ gammaMatrix(j, gamma, theta, n_o_list, n_e_list)


def transferMatrix(gamma, d_list, n_o_list, n_e_list, lam_vac, theta):

    #  convert lists to numpy arrays if they're not already.
    d_list = np.array(d_list, dtype=float)
    n_o_list = np.array(n_o_list)
    n_e_list = np.array(n_e_list)

    # input tests
    # if hasattr(lam_vac, 'size') and lam_vac.size > 1:
    #     raise ValueError('This function is not vectorized; you need to run one '
    #                      'calculation at a time (1 wavelength, 1 angle, etc.)')
    # if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
    #     raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if gamma != (0 or 1):
        raise ValueError('Gamma must be 0 or 1!')

    num_layers = np.size(d_list)

    systemMatrix = np.zeros((num_layers, 2, 2), dtype=complex)
    for i in range(num_layers-2, 0, -1):
        systemMatrix = systemMatrix @ layerMatrix


if __name__ == "__main__":

    gamma = 1                   # TE (0) or TM (1)
    d_list = [inf, 1000, inf]   # in nm
    n_o_list = [1, 2, 1]        # for TE mode
    n_e_list = [1, 2, 1]        # for TM mode
    lam_vac = 1540
    theta = pi/4

    transferMatrix(gamma, d_list, n_o_list, n_e_list, lam_vac, theta)
