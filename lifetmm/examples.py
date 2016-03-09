from __future__ import division, print_function, absolute_import

# from .lifetmm_core import *
from lifetmm import *
from tqdm import *
from numpy import pi, linspace, inf, array, sum, cos, sin
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# To run a sample use the following in python console:

# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180
mmTOnm = 1E6


def mcgehee():
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
            structure.transfer_matrix()
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


def mcgehee2():
    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252]
    # list of wavelengths to evaluate
    lambda_vac = 600

    structure = LifetimeTmm(d_list, n_list)
    structure.setWavelength(lambda_vac)

    data = np.zeros(sum(d_list[1:-1]))
    for th in [0]:
        for pol in ['s']:
            structure.setPolarization(pol)
            structure.setAngle(th)
            structure.transfer_matrix()
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


def vrendenbergen():
    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252+7.25439j]
    # list of wavelengths to evaluate
    lambda_vac = 600
    # Doped layer (mote array index starts at zero)
    m = 3
    # Bulk refractive index
    n_a = n_list[m]

    structure = LifetimeTmm(d_list, n_list)
    structure.setActiveLayer(m)
    structure.setWavelength(lambda_vac)
    structure.setBulkRefract(n_a)

    data = np.zeros(sum(d_list[1:-1]))
    for th in [0]:
        for pol in ['s']:
            structure.setPolarization(pol)
            structure.setAngle(th)
            structure.vrendenberg()
            # structure.transfer_matrix()
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


def test():
    # # list of layer thicknesses in nm
    # d_list = [inf, 1000, 50, 50, 100, 25, 50, 50, 1000, inf]
    # # list of refractive indices
    # n_list = [1, 1, 3, 1, 1.54, 1, 3, 1, 3, 3]
    # # Doped layer (mote array index starts at zero)
    # m = 4

    # list of layer thicknesses in nm
    d_list = [inf, 250, 500, 1000, inf]
    # list of refractive indices
    n_list = [1.54, 1.54, 1.54, 1.54, 1.54]
    # Doped layer (mote array index starts at zero)
    m = 2

    # list of free space wavelengths to evaluate
    lambda_vac = 1550
    # Bulk refractive index
    n_a = n_list[m]

    structure = LifetimeTmm(d_list, n_list)
    structure.set_active_layer(m)
    structure.set_wavelength(lambda_vac)
    structure.set_bulk_n(n_a)

    # structure.show_structure()
    #
    # structure.set_angle(0)
    # structure.set_polarization('s')
    # x, E_square = structure.layer_E_Field(m)
    # x, E_square = structure.transfer_matrix()
    # plt.figure()
    # plt.plot(x, E_square)
    # dsum = getattr(structure, 'd_cumsum')
    # plt.axhline(y=1, linestyle='--', color='k')
    # for i, xmat in enumerate(dsum):
    #     plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
    # plt.xlabel('Position in Device (nm)')
    # plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    # plt.legend(title='Theta')
    # plt.show()

    # # Evaluate using scipy's integrate function one x at a time
    # Ez = []
    # for x in tqdm(range(d_list[m])):
    #     # print('Evaluating at x = {:.2f}'.format(x))
    #     Ez.append(structure.purcell_factor_z(x))
    # plt.figure()
    # plt.plot(Ez)
    # plt.show()

    # Evaluate for entire layer at once
    result = structure.purcell_factor_layer()
    plt.figure()
    plt.plot(result)
    plt.show()


if __name__ == "__main__":
    # mcgehee()
    test()
    # vrendenbergen()

