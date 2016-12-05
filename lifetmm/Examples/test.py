import matplotlib.pyplot as plt

from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm
from lifetmm.Methods.TransferMatrix import TransferMatrix


def mcgehee():
    st = TransferMatrix()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704 + 0.01161j)
    st.add_layer(35, 1.4621 + 0.04426j)
    st.add_layer(220, 2.12 + 0.3166016j)
    st.add_layer(7, 2.095 + 2.3357j)
    st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    st.set_vacuum_wavelength(600)
    st.set_polarization('s')
    st.set_field('E')
    st.set_incident_angle(0, units='degrees')
    st.info()

    # Do calculations
    result = st.calc_field_structure()
    z = result['z']
    y = result['field_squared']

    # Plot results
    plt.figure()
    plt.plot(z, y)
    for z in st.get_layer_boundaries()[:-1]:
        plt.axvline(x=z, color='k', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$ Intensity ($|E(z)/E_0(0)|^2$)')
    if SAVE:
        plt.savefig('../Images/McGehee structure.png', dpi=300)
    plt.show()


def spe():
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)

    # Add layers
    # st.add_layer(lam0, 1)
    st.add_layer(lam0, si)
    st.add_layer(lam0, air)
    st.add_layer(lam0, si)
    # st.add_layer(lam0, 1)

    # Get results
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']
    spe_TE = spe['TE_total']
    spe_TM_p = spe['TM_p_total']
    spe_TM_s = spe['TM_s_total']

    # Plot spe rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe_TE, label='TE')
    ax1.plot(z, spe_TM_p, label='TM')
    ax1.plot(z, spe_TE + spe_TM_p, 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe_TM_s, label='TM')

    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('Position in layer (nm)')

    ax1.axhline(y=1, linestyle='--', color='k')
    ax2.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(z, color='k', lw=2)
        ax2.axvline(z, color='k', lw=2)
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')
    plt.show()


def guiding_plot():
    """ Find the guiding modes (TE and TM) for a given structure.
    First plot s_11 as a function of beta. When s_11=0 this corresponds
    to a wave guiding mode. We then solve the roots (with scipy's brentq
    algorithm) and plot these as vertical red lines. Check that visually there
    is a red line at each pole so that none are missed.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.set_field('E')
    st.set_leaky_or_guiding('guiding')

    # st.add_layer(0 * lam0, air)
    # st.add_layer(1 * lam0, si)
    # st.add_layer(0 * lam0, air)

    st.add_layer(300, sio2)
    st.add_layer(100, si)
    st.add_layer(20, sio2)
    st.add_layer(100, si)
    st.add_layer(300, air)

    st.info()

    # Prepare the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')

    # TE modes
    st.set_polarization('TE')
    [beta, s_11] = st.s11_guided()
    ax1.plot(beta, s_11, label='TE')
    roots = st.calc_guided_modes(normalised=True)
    for root in roots:
        ax1.axvline(root, color='r')

    # TM modes
    st.set_polarization('TM')
    [beta, s_11] = st.s11_guided()
    ax2.plot(beta, s_11, label='TM')
    roots = st.calc_guided_modes(normalised=True)
    for root in roots:
        ax2.axvline(root, color='r')

    # Format plot
    # fig.tight_layout()
    ax1.set_ylabel('$S_{11}$')
    ax1.axhline(color='k')
    ax2.set_ylabel('$S_{11}$')
    ax2.set_xlabel('Normalised parallel wave vector (k_11/k)')
    ax2.axhline(color='k')
    ax1.legend()
    ax2.legend()
    if SAVE:
        plt.savefig('../Images/guided modes.png', dpi=300)
    plt.show()


def guiding_electric_field():
    # Create structure
    st = TransferMatrix()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1.5 * lam0, air)
    st.add_layer(lam0, si)
    st.add_layer(1.5 * lam0, air)
    st.info()

    st.set_polarization('TM')
    st.set_field('H')
    st.set_leaky_or_guiding('guiding')
    alpha = st.calc_guided_modes(normalised=True)
    plt.figure()
    for i, a in enumerate(alpha):
        st.set_guided_mode(a)
        result = st.calc_field_structure()
        z = result['z']
        # z = st.calc_z_to_lambda(z)
        E = result['field']
        # Normalise fields
        # E /= max(E)
        plt.plot(z, abs(E) ** 2, label=i)

    for z in st.get_layer_boundaries()[:-1]:
        # z = st.calc_z_to_lambda(z)
        plt.axvline(x=z, color='k', lw=1, ls='--')
    plt.legend(title='Mode index')
    if SAVE:
        plt.savefig('../Images/guided fields.png', dpi=300)
    plt.show()


def test():
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    # st.add_layer(1e3, si)
    st.add_layer(1900, sio2)
    st.add_layer(100, si)
    st.add_layer(20, sio2)
    st.add_layer(100, si)
    # st.add_layer(1900, sio2)
    st.add_layer(1e3, air)
    st.info()

    st.set_polarization('TE')
    st.set_field('E')
    st.set_leaky_or_guiding('guiding')
    alpha = st.calc_guided_modes(normalised=True)
    st.set_guided_mode(alpha[0])
    result = st.calc_field_structure()
    z = result['z']
    z = st.calc_z_to_lambda(z)
    E = result['field']
    # Normalise fields
    # E /= max(E)

    plt.figure()
    plt.plot(z, abs(E) ** 2)
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        plt.axvline(x=z, color='k', lw=1, ls='--')
    plt.show()

if __name__ == "__main__":
    SAVE = False

    # Set vacuum wavelength
    lam0 = 1550

    # Material refractive index at lam0
    air = 1
    sio2 = 1.45
    si = 3.48

    # mcgehee()
    # spe()
    # guiding_plot()
    # guiding_electric_field()
    test()
