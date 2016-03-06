from __future__ import division, print_function, absolute_import

# from .lifetmm_core import *
from lifetmm import *

from numpy import pi, linspace, inf, array, sum, cos, sin
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# To run a sample use the following in python console:

# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180
mmTOnm = 1E6


def mcgehee2():
    '''
    Copy of the stanford group simulation at 600nm
    '''

    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252+7.25439j]
    # list of wavelengths to evaluate
    lambda_vac = 600

    structure = LifetimeTmm(d_list, n_list)
    structure.setWavelength(lambda_vac)

    data = np.zeros(sum(d_list[1:-1]))
    for th in [0]:
        for pol in ['s']:
            structure.setPolarization(pol)
            structure.setAngle(th)
            structure.calculate()
            data = getattr(structure, 'E_square')

    plt.figure()
    plt.plot(data, label='data')
    dsum = getattr(structure, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def vrendenberg():
    d_sio2 = 270
    d_si = 115
    n_si = 3.4
    n_sio2 = 1.44

    # list of layer thicknesses in nm
    d_list = [inf, 2000, d_sio2, d_si, d_sio2, d_si, d_sio2,
              540,
              d_sio2,  d_si, d_sio2,  d_si, d_sio2,  d_si, d_sio2,  d_si, 2000, inf]
    # list of refractive indices
    n_list = [1, 1, n_sio2, n_si, n_sio2, n_si, n_sio2,
              1.54,
              n_sio2,  n_si, n_sio2,  n_si, n_sio2,  n_si, n_sio2,  n_si, 1, 1]
    # list of wavelengths to evaluate in nm
    lambda_vac = 1600

    structure = LifetimeTmm(d_list, n_list)
    structure.setWavelength(lambda_vac)

    plt.figure()
    data = np.zeros(sum(d_list[1:-1]))

    theta_max = thetaCritical(m, n_list)
    for th in [0]:
        structure.setAngle(th)
        n_m = n_list[7]
        th_1 = snell(n_m, n_list[0], th)
        th_s = snell(n_m, n_list[-1], th)

        for pol in ['p', 's']:
            structure.setPolarization(pol)

            structure.calculate()
            data = getattr(structure, 'E_square')
            plt.plot(data, label=pol)

    dsum = getattr(structure, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i in dsum:
        plt.axvline(x=i, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def test():

    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704, 1.4621, 2.12, 2.095, 1.20252]
    # Doped layer
    m = 3
    # list of wavelengths to evaluate
    lambda_vac = 600
    # Bulk refractive index
    n_a = 1.5

    structure = LifetimeTmm(d_list, n_list)
    structure.setActiveLayer(m)
    structure.setWavelength(lambda_vac)

    plt.figure()
    data = np.zeros(sum(d_list[1:-1]))

    th_critical = thetaCritical(m, n_list)
    th_emission = np.linspace(0, th_critical, 50, endpoint=False)

    for pol in ['s', 'p']:
        structure.setPolarization(pol)
        for th_m in th_emission:
            # Corresponding angle out of structure from superstrate
            th_1 = snell(n_list[m], n_list[0], th_m)
            structure.setAngle(th_1)
            # Evaluate E(x)**2 inside structure
            structure.calculate()
            data += getattr(structure, 'E_square')

            # Corresponding angle out of structure from substrate
            th_s = snell(n_list[m], n_list[-1], th_m)
            structure.setAngle(th_s)
            # Evaluate E(x)**2 inside structure
            structure.flip()
            structure.calculate()
            structure.flip()
            data += getattr(structure, 'E_square')

            # Evaluate integral
            first_term = (3/(2*n_a)) * Mj2(j, th_1, th_s, th_m) * (abs(U_j)**2 / 3)
            n_list[m]**2 * cos(th_m) * sin(th_m)

    plt.plot(data, label=th_m)

    dsum = getattr(structure, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend(title='Theta')
    plt.show()


if __name__ == "__main__":
    # mcgehee2()
    test()
    # vrendenberg()

