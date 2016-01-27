from numpy import cos, sin, inf, zeros, exp, conj, nan, isnan, pi, arcsin
import numpy as np
import matplotlib.pyplot as plt
# from .lifetmm_core import *
from lifetmm import *

def Mj2(j, theta_1, theta_s,theta):
    n_1 = n_list[0]
    n_s = n_list[-1]

    E_field_s = TransferMatrix(d_list, n_list, lam_vac=1540, th_0=theta * degree, pol='s', reverse=False)['E_avg']
    E_field_p = TransferMatrix(d_list, n_list, lam_vac=1540, th_0=theta * degree, pol='p', reverse=False)['E_avg']

    E_1 = (E_field_p[1] + E_field_s[1]) / 2
    E_s = (E_field_p[-2] + E_field_s[-2]) / 2


    denom = eps0 * (n_1*H(theta_1)*abs(E_1**2) + n_s*H(theta_s)*abs(E_s**2))
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

    # Using Snell's law evaluate the critical angle
    theta_crit = arcsin(n_clad/n_list[m])

    return max(theta_crit, pi/2)


def sumAngles(j, n_list, m):
    result = 0
    theta_max = np.linspace(0, degree*thetaCritical(m, n_list))
    for theta in theta_max:
        # Use snell's law to evaluate theta_1 and theta_s for theta in the active layer
        n_1 = n_list[1]         # Ambient refractive index
        n_s = n_list[-2]        # Substrate refractive index
        n_m = n_list[m]         # Film refractive index
        theta_1 = arcsin(n_m*sin(theta)/n_1)
        theta_s = arcsin(n_m*sin(theta)/n_s)

        E_field_s = TransferMatrix(d_list, n_list, lam_vac=1540, th_0=theta * degree, pol='s', reverse=False)['E']
        E_field_p = TransferMatrix(d_list, n_list, lam_vac=1540, th_0=theta * degree, pol='p', reverse=False)['E']

        U_j = (E_field_p + E_field_s) / 2

        First_term = (3/(2*n_a)) * eps0 * Mj2(j, theta_1, theta_s,theta) * abs(U_j)**2 / 3

        H_term = (H(theta_1) + H(theta_s)) / (H(theta_1)*cos(theta_1) + H(theta_s)*cos(theta_s))
        weighting = (n_m**2) * cos(theta) * sin(theta)
        result += First_term * H_term * weighting
        return result


degree = pi / 180
lam = 1540

# list of layer thicknesses in nm
d_list = [inf, 1000, 1000, inf]
# list of refractive indices
n_list = [1.5, 1.5, 3, 3]

m = 3                   # Layer
n_a = 1.54              # Bulk refractive index
eps0 = 8.854187817E-12  # Vacuum permitivity (F.m^-1)

enhancement = 0
# Sum over modes
for j in range(1, 5):
    enhancement += (1/2) * sumAngles(j, n_list, m)

plt.figure()
plt.plot(enhancement)
plt.show()
