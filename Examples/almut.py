"""
Calculations to design structure to match Nik and Almut's paper.
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.TransferMatrix import TransferMatrix


def dbr_reflectivity():
    """ Reflection coefficient for top and bottom DBRs (at 1.55um) given in [1]"""

    # Emission wavelength (nm)
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442
    edts = 1.6

    # Setup structure
    st = TransferMatrix()
    st.add_layer(100, sio2)
    st.add_layer(lam0 / (4 * sio2), sio2)
    st.add_layer(lam0 / (4 * si), si)
    st.add_layer(lam0 / (2 * sio2), sio2)

    st.set_vacuum_wavelength(lam0)
    st.set_incident_angle(0, units='degrees')
    st.info()
    st.show_structure()

    r, t = st.calc_reflection_and_transmission(correction=False)
    print('Bottom DBR: R={0:.3f}% and T={1:.3f}%'.format(100 * r, 100 * t))


def dbr_reflectivity_vs_aoi():
    """ Reflection coefficient for Fabry-Perot microcavity (at 1.55um) vs AOI"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    st.add_layer(100, sio2)
    st.add_layer(lam0 / (4 * sio2), sio2)
    st.add_layer(lam0 / (4 * si), si)
    st.add_layer(lam0 / (2 * sio2), sio2)

    st.set_vacuum_wavelength(lam0)
    # st.show_structure()
    st.info()

    th_list = np.linspace(0, 90, 2000, endpoint=False)
    rs_list = []
    rp_list = []
    for theta in th_list:
        # Do calculations
        st.set_incident_angle(theta, units='degrees')
        st.set_polarization('s')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rs_list.append(r)
        st.set_polarization('p')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rp_list.append(r)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(th_list, rs_list, '--', label='s')
    ax.plot(th_list, rp_list, label='p')
    ax.set_xlabel('AOI (degrees)')
    ax.set_ylabel(r'Reflection ($|r|^2)$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # dbr_reflectivity()
    dbr_reflectivity_vs_aoi()
