"""
Example scripts for the LifetimeTmm package.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lifetmm.SPE import SPE


def example1():
    """ SE rate for Silicon to air semi-infinite half spaces."""
    from lifetmm.SPE import SPE
    import pandas as pd
    from lifetmm.Materials import n_1540nm as sample
    """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""
    # Load Sample Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Create Structure
    st = SPE()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(chip['d'], chip['n'])
    st.add_layer(d_clad * lam0, n_dict['Air'])
    st.info()

    # Calculate
    res = st.calc_spe_structure(th_pow=11)
    z = res['z']
    z = st.calc_z_to_lambda(z)

    # ------- Plots -------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, res['leaky']['avg'])
    if st.supports_guiding():
        ax2.plot(z, res['guided']['avg'])

    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$)')
    ax1.set_title('Leaky')
    ax2.set_title('Guided')

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
        plt.savefig('../Images/{}_individual'.format(chip['Sample ID']))

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
        plt.savefig('../Images/{}_total'.format(chip['Sample ID']))
    plt.show()


def example2():
    """Calculate the fresnel reflection/transmission from (multilayer) planar interfaces
    as a function of incidence angle."""
    from lifetmm.TransferMatrix import TransferMatrix
    from lifetmm.Materials import n_1540nm

    lam0 = 1550  # Vacuum wavelength

    # Setup simulation
    st = TransferMatrix()
    st.add_layer(0, n_1540nm['SiO2'])
    st.add_layer(2, n_1540nm['Ti'])
    st.add_layer(10, n_1540nm['Au'])
    st.add_layer(2, n_1540nm['Ti'])
    st.add_layer(0, n_1540nm['SiO2'])
    st.set_vacuum_wavelength(lam0)
    st.info()

    # Plot and save figs
    res = st.calc_reflectivity_vs_angle()
    res['fig'].savefig('../Images/fresnel/reflectivity_vs_angle', dpi=900)
    res = st.calc_transmission_vs_angle()
    res['fig'].savefig('../Images/fresnel/transmission_vs_angle', dpi=900)

    # Save structure to .txt file
    with open('../Images/fresnel/structure.txt', 'w') as the_file:
        the_file.write('Free space wavelength: {}\n\n'.format(st.lam_vac))
        the_file.write('d,n\n')
        for n, d in zip(st.n_list, st.d_list):
            the_file.write('{:.4g},{:.4g}\n'.format(d, n))


def example3():
    """ SE rate for Silicon to air semi-infinite half spaces."""
    # """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""
    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(0.5 * lam0, 1.45)
    st.add_layer(100, 1.65)
    st.add_layer(500, 1.45)
    st.add_layer(0.5 * lam0, 1)
    st.set_vacuum_wavelength(lam0)
    st.info()

    # Calculate
    res = st.calc_spe_structure(th_pow=11)
    z = res['z']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # ------- Plots -------
    # Plot data
    fig, ax1 = plt.subplots()
    if st.supports_guiding():
        ax1.plot(z, res['leaky']['avg'] + res['guided']['avg'], label='Avg')
        ax1.plot(z, res['leaky']['parallel'] + res['guided']['parallel'], '--', label=r'$\parallel$')
        ax1.plot(z, res['leaky']['perpendicular'] + res['guided']['perpendicular'], '-.', label=r'$\bot$')
    else:
        ax1.plot(z, res['leaky']['avg'], label='Avg')
        ax1.plot(z, res['leaky']['parallel'], '--', label=r'$\parallel$')
        ax1.plot(z, res['leaky']['perpendicular'], '-.', label=r'$\bot$')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z [$\lambda$]')
    ax1.legend(fontsize='small')
    ax1.set_ylim(0, ax1.get_ylim()[1])
    bounds = ax1.get_ylim()

    # Draw rectangles for the refractive index
    ax2 = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax2.set_ylabel('n')
    ax2.set_ylim(bounds)

    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    for zb in st.get_layer_boundaries()[:-1]:
        zb = st.calc_z_to_lambda(zb)
        ax1.axvline(x=zb, color='k', lw=2)

    ax1.set_xlim([min(z), max(z)])

    if SAVE:
        plt.savefig('../Images/Air')
    plt.show()


if __name__ == "__main__":
    SAVE = False
    # example1()
    # example2()
    example3()
