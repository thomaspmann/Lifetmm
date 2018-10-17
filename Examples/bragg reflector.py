"""
Calculate the fresnel reflection from Fabry-Perot microcavities as a function of incidence angle.

i.e. two distributed Bragg reflectors (DBR) cladding an active region of lam0/2 thickness.

[1] Erbium implantation in optical microcavities for controlled spontaneous emission, Vredenberg
"""

from lifetmm.TransferMatrix import TransferMatrix


def dbr_reflectivity():
    """ Reflection coefficient for top and bottom DBRs (at 1.55um) given in [1]"""

    # Emission wavelength
    lam0 = 1550
    # Material refractive index @ lam0
    si = 3.48
    sio2 = 1.442

    # Setup simulation
    st = TransferMatrix()
    # Bottom DBR
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

    r, t = st.calc_reflectance_and_transmittance(correction=False)
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
    r, t = st.calc_reflectance_and_transmittance(correction=False)
    print('Top DBR: R={0:.3f}% and T={1:.3f}%'.format(100 * r, 100 * t))

    # Whole DBR
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
    # Reflection coefficient for Fabry-Perot microcavity (at 1.55um) vs AOI
    st.calc_reflectivity_vs_angle()
    # Reflection coefficient for Fabry-Perot microcavity (at 1.55um) vs wavelength.
    # Drop in refletance gives the resonance wavelength of the cavity.
    st.calc_reflectivity_vs_wavelength(1500, 1600)

if __name__ == "__main__":
    dbr_reflectivity()
