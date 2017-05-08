"""
Calculate the fresnel reflection from Fabry-Perot microcavities as a function of incidence angle.

i.e. two distributed Bragg reflectors (DBR) cladding an active region of lam0/2 thickness.

[1] Erbium implantation in optical microcavities for controlled spontaneous emission, Vredenberg
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.TransferMatrix import TransferMatrix


def dbr():
    """ Reflection coefficient for DBR (at 1.55um)"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(0, sio2)
    for i in range(2):
        st.add_layer(lam0 / 4, si)
        st.add_layer(lam0 / 4, sio2)
    st.add_layer(lam0 / 4, si)

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


def fabry_perot():
    """ Reflection coefficient for Fabry-Perot microcavity (at 1.55um)"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    st.set_vacuum_wavelength(lam0)

    # # Bottom
    # st.add_layer(0, sio2)
    # for i in range(4):
    #     st.add_layer(lam0/4, sio2)
    #     st.add_layer(lam0 / 4, si)
    #
    # # Active region
    # st.add_layer(lam0 / 2, sio2)

    # Top DBR
    for i in range(2):
        st.add_layer(lam0 / 4, si)
        st.add_layer(lam0 / 4, sio2)
    st.add_layer(lam0 / 4, si)
    st.add_layer(0, 1)

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
    # dbr()
    fabry_perot()
