from __future__ import division, print_function, absolute_import

# from .lifetmm_core import *
from lifetmm import *
from tqdm import *
from numpy import pi, linspace, inf, array, sum, cos, sin
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt

# # To run a sample use the following in python console:
# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180
mmTOnm = 1E6


def mcgehee():
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j)
    st.add_layer(220, 2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0, units='degrees')

    y = st.structure_E_field(time_reversal=False)['E_square']

    plt.figure()
    plt.plot(y)
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, zmat in enumerate(dsum):
        plt.axvline(x=zmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def spe():
    st = LifetimeTmm()

    st.add_layer(0, 3.48)
    st.add_layer(2000, 3.48)
    st.add_layer(200, 1)
    st.add_layer(2000, 5)
    st.add_layer(2000, 1)
    st.add_layer(0, 1)

    st.set_wavelength(1550)
    st.set_polarization('s')

    result = st.spe_structure()
    # result = st.spe_rate_structure()
    y = result['spe']

    z = result['z']

    plt.figure()
    plt.plot(z, y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, zmat in enumerate(dsum):
        plt.axvline(x=zmat, linestyle='-', color='r', lw=2)
    plt.show()


def test():
    st = LifetimeTmm()

    st.add_layer(0, 3.48)
    st.add_layer(2000, 3.48)
    st.add_layer(2000, 1)
    st.add_layer(0, 1)
    # st.add_layer(0, 3.48)

    st.set_wavelength(1550)
    st.set_polarization('s')
    st.set_angle(70, units='degrees')

    y = st.structure_E_field(time_reversal=True)['E_square']

    plt.figure()
    plt.plot(y)
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, zmat in enumerate(dsum):
        plt.axvline(x=zmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


if __name__ == "__main__":
    # mcgehee()
    # test()
    spe()
