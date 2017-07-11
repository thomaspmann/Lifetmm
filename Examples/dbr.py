import numpy as np

from lifetmm.TransferMatrix import TransferMatrix


def example1():
    """
    Recreate calculations and plots in https://www.photonics.com/EDU/Handbook.aspx?AID=25501
    """

    # Emission wavelength
    lam0 = 850
    # n @ lam0
    n_h = 2.5
    n_l = 1.46
    n_s = 1.52
    # Number of n_l layers (n_h has p+1 layers)
    p = 5

    def n_e():
        return pow(n_h, 2 * p + 2) / (n_s * pow(n_l, 2 * p))

    def r():
        """Percentage reflectance"""
        ne = n_e()
        return 100 * pow(((1 - ne) / (1 + ne)), 2)

    def lam_edge():
        delta = 1 / np.deg2rad(90) * np.arcsin((n_h - n_l) / (n_h + n_l))
        return lam0 / (1 + delta), lam0 / (1 - delta)

    print(r(), lam_edge())

    st = TransferMatrix()
    st.add_layer(0, n_s)
    for i in range(p):
        st.add_layer(lam0 / (n_h * 4), n_h)
        st.add_layer(lam0 / (n_l * 4), n_l)
    st.add_layer(lam0 / (n_h * 4), n_h)

    st.set_vacuum_wavelength(lam0)
    # st.show_structure()
    st.info()
    st.plot_reflectivity_vs_wavelength()


def example2():
    """ Reflection coefficient for Fabry-Perot microcavity (designed at 1.55um) vs lam0"""
    # Emission wavelength
    lam0 = 1550
    # n @ lam0
    n_h = 2.3
    n_l = 1.43
    n_s = 1.6
    # Number of n_l layers (n_h has p+1 layers)
    p = 6

    # Setup simulation
    st = TransferMatrix()
    st.add_layer(0, n_s)
    for i in range(p):
        st.add_layer(lam0 / (n_h * 4), n_h)
        st.add_layer(lam0 / (n_l * 4), n_l)
    st.add_layer(lam0 / (n_h * 4), n_h)

    st.set_vacuum_wavelength(lam0)
    # st.show_structure()
    st.info()
    st.plot_reflectivity_vs_angle()


if __name__ == "__main__":
    example1()
    example2()
