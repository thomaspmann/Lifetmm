import matplotlib.pyplot as plt
import numpy as np

from lifetmm.SpontaneousEmissionRate import LifetimeTmm
from lifetmm.TransferMatrix import TransferMatrix


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
        plt.savefig('../Images/McGehee structure')
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
        st.set_mode_n_11(a)
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
    # Dictionary of material refractive indexes
    n_dict = {'Air': 1,
              'Water': 1.3183,
              'SiO2': 1.442,
              'Glycerol': 1.46,
              'EDTS': 1.56,
              'Cassia Oil': 1.6,
              'Diiodomethane': 1.71,
              'TZN': 1.9
              }

    # Multilayer Structure
    d_clad = 1.5
    st = TransferMatrix()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(1 * lam0, n_dict['EDTS'])
    st.add_layer(d_clad * lam0, n_dict['Air'])
    # st.flip()
    st.info()

    # Simulation Wavelength
    st.set_vacuum_wavelength(1550)
    st.set_field('E')
    st.set_polarization('TE')

    # Find guided mode indices
    st.set_leaky_or_guiding('guiding')
    alpha = st.calc_guided_modes(normalised=True)

    fig, ax1 = plt.subplots()
    for a in alpha:
        st.set_mode_n_11(a)
        result = st.calc_field_structure()
        z = result['z']
        E = result['field']
        ax1.plot(z, abs(E) ** 2, label='No absorption')

    # Multilayer Structure - now with absorption
    d_clad = 1.5
    st = TransferMatrix()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(1 * lam0, n_dict['EDTS'])
    st.add_layer(d_clad * lam0, n_dict['Air']+1j)
    st.info()
    st.flip()

    # Simulation Wavelength
    st.set_vacuum_wavelength(1550)
    st.set_field('E')
    st.set_polarization('TE')

    # Find guided mode indices
    st.set_leaky_or_guiding('guiding')
    alpha = st.calc_guided_modes(normalised=True)

    for a in alpha:
        st.set_mode_n_11(a)
        result = st.calc_field_structure()
        z = result['z']
        E = result['field']
        ax1.plot(z, abs(E) ** 2, label='Absorption')

        # Draw rectangles for the refractive index
    from matplotlib.patches import Rectangle
    ax2 = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        # z0 = st.calc_z_to_lambda(z0)
        # dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax2.set_ylabel('n')
    ax2.set_ylim(0, 1.5 * max(st.n_list.real))
    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(x=z, color='k', lw=2)

    ax1.legend()
    if SAVE:
        plt.savefig('../Images/guided fields.png', dpi=300)
    plt.show()


def PJ_dipoe_calculation():
    from matplotlib.patches import Rectangle

    st = LifetimeTmm()
    st.set_vacuum_wavelength(514.5)

    # Add layers
    # st.add_layer(lam0, 1.44)
    st.add_layer(750, 1 ** 0.5)
    # st.add_layer(750, 1)
    st.add_layer(750, 2.3409 ** 0.5)
    st.info()

    result = st.calc_spe_structure(th_pow=11)
    try:
        # spe = result['leaky']['perpendicular'] + result['guided']['perpendicular']
        spe = result['leaky']['parallel'] + result['guided']['parallel']
    except KeyError:
        # spe = result['leaky']['perpendicular']
        spe = result['leaky']['parallel']

    z = result['z']

    fig, ax1 = plt.subplots()
    ax1.plot(z, spe)

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z (nm)')
    # ax1.legend()

    # Draw rectangles for the refractive index
    ax2 = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax2.set_ylabel('n')
    ax2.set_ylim(ax1.get_ylim())
    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    for zb in st.get_layer_boundaries()[:-1]:
        ax1.axvline(x=zb, color='k', lw=2)

    if SAVE:
        plt.savefig('../Images/PJ_dipole')
    plt.show()


def tester():
    """Plot reflection from a structure vs z component wave-vector"""
    st = LifetimeTmm()
    st.set_vacuum_wavelength(514.5)
    st.set_polarization('TM')

    # Add layers
    st.add_layer(0, 1)
    st.add_layer(0, 2 ** 0.5)
    # st.add_layer(0, (-8+0.96j)**0.5)
    st.info()

    n_11_list = np.linspace(0, 5, 2000)
    r_list = []
    for n_11 in n_11_list:
        st.n_11 = n_11
        reflection, transmission = st.calc_reflection_and_transmission()
        r_list.append(np.sqrt(reflection))

    fig, ax = plt.subplots()
    ax.plot(n_11_list, r_list)
    # Line to show plane/evanescent wave boundary
    ax.axvline(1, color='k')
    ax.set_xlabel(r'$k_{z}/k_{0}$')
    ax.set_ylabel(r'$|r|^2$')
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
    # PJ_dipoe_calculation()
    # tester()
