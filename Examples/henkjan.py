import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lifetmm.SpontaneousEmissionRate import LifetimeTmm


def PJ_dipoe_calculation():
    st = LifetimeTmm()
    # st.add_layer(lam0, 1.44)
    st.add_layer(750, 1 ** 0.5)
    # st.add_layer(750, 1)
    st.add_layer(750, 2.3409 ** 0.5)
    st.set_vacuum_wavelength(514.5)
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


def plot_vertical_horizontal_total():
    """Plot SE rates for vertical and horizontal dipoles. Then plot sum for randomly orientated dipole."""

    # Create structure
    st = LifetimeTmm()
    st.add_layer(2.5 * lam0, air)
    st.add_layer(lam0, si)
    st.add_layer(2.5 * lam0, air)
    st.set_vacuum_wavelength(lam0)
    st.info()

    # Calculate
    res = st.calc_spe_structure(th_pow=11)
    z = res['z']
    z = st.calc_z_to_lambda(z)

    # ------- Plots -------
    # leaky modes plot
    spe = res['leaky']
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s'], label='TM')

    ax1.set_ylim(0, 1.4)
    ax2.set_ylim(0, 1.4)
    ax1.set_title('Spontaneous Emission Rate (leaky). Silicon (n=3.48) with air cladding (n=1.)')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    if SAVE:
        plt.savefig('../Images/henkjan_leaky')
    plt.show()

    # Guided modes plot
    spe = res['guided']
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s'], label='TM')

    ax1.set_title('Spontaneous Emission Rate (guided). Silicon (n=3.48) with air cladding (n=1.)')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    if SAVE:
        plt.savefig('../Images/henkjan_guided')
    plt.show()

    # parallel and perpendicular dipole orientation (leaky + guided)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    if st.supports_guiding():
        ax1.plot(z, res['leaky']['parallel'] + res['guided']['parallel'], label='h')
        ax2.plot(z, res['leaky']['perpendicular'] + res['guided']['perpendicular'], label='v')
    else:
        ax1.plot(z, res['leaky']['parallel'], label='h')
        ax2.plot(z, res['leaky']['perpendicular'], label='v')

    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$)')
    ax1.set_title('Parallel/Horizontal')
    ax2.set_title('Perpendicular/Vertical')

    for zb in st.get_layer_boundaries()[:-1]:
        zb = st.calc_z_to_lambda(zb)
        ax1.axvline(x=zb, color='k', lw=2)
        ax2.axvline(x=zb, color='k', lw=2)

    # Draw rectangles for the refractive index
    ax1b = ax1.twinx()
    ax2b = ax2.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.2)
        ax1b.add_patch(rect)
        ax1b.set_ylabel('n')
        ax1b.set_ylim(0, 1.5 * max(st.n_list.real))
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.2)
        ax2b.add_patch(rect)
        ax2b.set_ylabel('n')
        ax2b.set_ylim(0, 1.5 * max(st.n_list.real))
    ax1.set_zorder(ax1b.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'
    ax2.set_zorder(ax2b.get_zorder() + 1)  # put ax1 in front of ax2
    ax2.patch.set_visible(False)  # hide ax1'canvas'

    if SAVE:
        plt.savefig('../Images/{henkjan}_individual')

    # Average dipole orientation (leaky + guided)
    fig, ax1 = plt.subplots()
    if st.supports_guiding():
        ax1.plot(z, res['leaky']['avg'] + res['guided']['avg'], label='Avg')
    else:
        ax1.plot(z, res['leaky']['avg'], label='Avg')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    ax1.legend()

    # Draw rectangles for the refractive index
    ax2 = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax2.set_ylabel('n')
    ax2.set_ylim(0, 1.5 * max(st.n_list.real))
    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    for zb in st.get_layer_boundaries()[:-1]:
        zb = st.calc_z_to_lambda(zb)
        ax1.axvline(x=zb, color='k', lw=2)

    if SAVE:
        plt.savefig('../Images/henkjan_total')
    plt.show()


if __name__ == "__main__":
    SAVE = True

    # Set vacuum wavelength
    lam0 = 1550

    # Material refractive index at lam0
    air = 1
    sio2 = 1.45
    si = 3.48

    PJ_dipoe_calculation()
    # plot_vertical_horizontal_total()
