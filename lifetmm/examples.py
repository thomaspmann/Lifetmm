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
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j, active=True)
    st.add_layer(220,2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    st.add_layer(0, 1.20252+7.25439j)

    st.show_structure()

    st.set_wavelength(600)
    st.set_bulk_n(1.54)
    st.set_angle(0)
    st.set_polarization('s')
    x, y = st.structure_E_field(result='E_square')

    plt.figure()
    plt.plot(x, y)
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def purcell_layer():
    st = LifetimeTmm()
    Er = 1.5
    lam = 1540
    st.add_layer(0, 1)
    # st.add_layer(500, Er)
    st.add_layer(4*lam/(2*pi), Er, active=True)
    # st.add_layer(2000,3)
    st.add_layer(0, Er)

    st.set_wavelength(lam)
    st.set_bulk_n(Er)

    # st.show_structure()
    y = st.purcell_factor_layer()

    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.show()


def purcell_z():
    st = LifetimeTmm()
    Er = 1.54
    d_active = 100
    st.add_layer(0, Er)
    st.add_layer(500, Er)
    st.add_layer(d_active, Er, active=True)
    st.add_layer(500, 300)
    st.add_layer(0, 300)

    st.set_wavelength(1500)
    st.set_bulk_n(Er)

    # Evaluate using scipy's integrate function one x at a time
    # dsum = getattr(st, 'd_cumsum')
    y = []
    for x in tqdm(range(d_active)):
        # print('Evaluating at x = {:.2f}'.format(x))
        y.append(st.purcell_factor_z(x))
    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.show()


def test():
    st = LifetimeTmm()
    Er = 1.5
    lam = 1540
    st.add_layer(0, Er)
    st.add_layer(2000, Er, active=True)
    st.add_layer(0, 1)

    st.set_wavelength(lam)
    st.set_bulk_n(Er)

    # st.show_structure()
    y = st.purcell_factor_layer()

    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.show()

if __name__ == "__main__":
    # mcgehee()
    # purcell_z()
    # purcell_layer()
    test()

