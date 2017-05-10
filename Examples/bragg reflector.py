"""
Calculate the fresnel reflection from Fabry-Perot microcavities as a function of incidence angle.

i.e. two distributed Bragg reflectors (DBR) cladding an active region of lam0/2 thickness.

[1] Erbium implantation in optical microcavities for controlled spontaneous emission, Vredenberg
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lifetmm.TransferMatrix import TransferMatrix


def dbr_reflectivity():
    """ Reflection coefficient for top and bottom DBRs (at 1.55um) given in [1]"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    # Bottom
    st.add_layer(0, si)
    for i in range(4):
        st.add_layer(lam0 / (4 * sio2), sio2)
        st.add_layer(lam0 / (4 * si), si)
    # Active region
    st.add_layer(lam0 / (2 * sio2), sio2)

    st.set_vacuum_wavelength(lam0)
    st.set_incident_angle(0, units='degrees')
    # st.show_structure()
    # st.info()

    r, t = st.calc_reflection_and_transmission(correction=False)
    print('Bottom DBR: R={0:.3f}% and T={1:.3f}%'.format(100 * r, 100 * t))

    # Setup simulation
    st = TransferMatrix()

    # Active region
    st.add_layer(lam0 / (2 * sio2), sio2)
    # Top DBR
    for i in range(2):
        st.add_layer(lam0 / (4 * si), si)
        st.add_layer(lam0 / (4 * sio2), sio2)
    st.add_layer(lam0 / (4 * si), si)
    st.add_layer(0, 1)

    st.set_vacuum_wavelength(lam0)
    st.set_incident_angle(0, units='degrees')
    # st.show_structure()
    # st.info()
    r, t = st.calc_reflection_and_transmission(correction=False)
    print('Top DBR: R={0:.3f}% and T={1:.3f}%'.format(100 * r, 100 * t))


def fabry_perot_vs_aoi():
    """ Reflection coefficient for Fabry-Perot microcavity (at 1.55um) vs AOI"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    # Bottom
    st.add_layer(0, sio2)
    for i in range(4):
        st.add_layer(lam0 / (4 * sio2), sio2)
        st.add_layer(lam0 / (4 * si), si)
    # Active region
    st.add_layer(lam0 / (2 * sio2), sio2)
    # Top DBR
    for i in range(2):
        st.add_layer(lam0 / (4 * si), si)
        st.add_layer(lam0 / (4 * sio2), sio2)
    st.add_layer(lam0 / (4 * si), si)
    st.add_layer(0, 1)

    st.set_vacuum_wavelength(lam0)
    st.show_structure()
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


def fabry_perot_vs_wavelength():
    """ Reflection coefficient for Fabry-Perot microcavity (at 1.55um) vs wavelength. 
    Drop in refletance gives the resonance wavelength of the cavity.
    """

    # Emission wavelength
    lam0 = 1550
    # Material refractive index @ lam0
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()

    # Bottom
    st.add_layer(0, sio2)
    for i in range(4):
        st.add_layer(lam0 / (4 * sio2), sio2)
        st.add_layer(lam0 / (4 * si), si)
    # Active region
    st.add_layer(lam0 / (2 * sio2), sio2)
    # Top DBR
    for i in range(2):
        st.add_layer(lam0 / (4 * si), si)
        st.add_layer(lam0 / (4 * sio2), sio2)
    st.add_layer(lam0 / (4 * si), si)
    st.add_layer(0, 1)

    st.set_vacuum_wavelength(lam0)
    st.set_incident_angle(0, units='degrees')
    # st.show_structure()
    st.info()

    lam_list = np.linspace(1500, 1600, 200, endpoint=False)
    rs_list = []
    rp_list = []
    for lam in tqdm(lam_list):
        # Do calculations
        st.set_vacuum_wavelength(lam)
        st.set_polarization('s')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rs_list.append(r)
        st.set_polarization('p')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rp_list.append(r)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(lam_list, rp_list, label='p')
    ax.plot(lam_list, rs_list, '--', label='s')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'Reflection ($|r|^2)$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dbr_reflectivity()
    fabry_perot_vs_aoi()
    fabry_perot_vs_wavelength()
