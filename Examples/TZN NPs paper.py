"""
Plots from

    'Photoluminescence of Femtosecond Laser Generated Er$^{3+}$
    Ion Doped Zinc-Sodium Tellurite Glass Nanoparticles'
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from lifetmm.SPE import SPE


def fig7():
    """ SE rate for Silicon to air semi-infinite half spaces."""
    # """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(0.05 * lam0, 3.48)
    st.add_layer(10, 2)
    st.add_layer(0.05 * lam0, 1)
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
    ax1.plot(z, res['leaky']['avg'], label='Avg')
    ax1.plot(z, res['leaky']['parallel'], '--', label=r'$\parallel$')
    ax1.plot(z, res['leaky']['perpendicular'], '-.', label=r'$\bot$')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
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

    if SAVE:
        plt.savefig('../Images/total')
    plt.show()


def supplementary_material1():
    """ SE rate for Silicon to air semi-infinite half spaces."""
    # """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    # WIDTH = 412.56  # the number (in pt) latex spits out when typing: \the\linewidth (paper 246, thesis 412.56)
    # FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy

    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(0.05 * lam0, 3.48)
    st.add_layer(10, 2)
    st.add_layer(0.05 * lam0, 1)
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
    ax1.plot(z, res['leaky']['avg'], label='Avg')
    ax1.plot(z, res['leaky']['parallel'], '--', label=r'$\parallel$')
    ax1.plot(z, res['leaky']['perpendicular'], '-.', label=r'$\bot$')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
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

    if SAVE:
        plt.savefig('../Images/SupplementaryMaterial1')
    plt.show()


def supplementary_material2():
    """ SE rate for Silicon to air semi-infinite half spaces."""
    # """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    # WIDTH = 412.56  # the number (in pt) latex spits out when typing: \the\linewidth (paper 246, thesis 412.56)
    # FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy

    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(0.6 * lam0, 3.48)
    st.add_layer(500, 2)
    st.add_layer(0.6 * lam0, 1)
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
    ax1.plot(z, res['leaky']['avg'], label='Avg')
    ax1.plot(z, res['leaky']['parallel'], '--', label=r'$\parallel$')
    ax1.plot(z, res['leaky']['perpendicular'], '-.', label=r'$\bot$')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
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

    if SAVE:
        plt.savefig('../Images/SupplementaryMaterial2')
    plt.show()


def further_work1():
    """
    Silicon to air semi-infinite half spaces.
    """

    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(lam0, 3.48)
    st.add_layer(50, 2)
    st.add_layer(lam0, 1)
    st.set_vacuum_wavelength(lam0)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(15, 7))
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], label='TE + TM')

    ax2.plot(z, spe['TE_lower_full'] + spe['TM_p_lower_full'], label='Fully radiative lower outgoing')
    ax2.plot(z, spe['TE_lower_partial'] + spe['TM_p_lower_partial'], label='Partially radiative lower outgoing')
    ax2.plot(z, spe['TE_upper'] + spe['TM_p_upper'], label='Fully radiative upper outgoing')

    ax3.plot(z, spe['TM_s'], label='TM')

    ax4.plot(z, spe['TM_s_lower_full'], label='Fully radiative lower outgoing')
    ax4.plot(z, spe['TM_s_lower_partial'], label='Partially radiative lower outgoing')
    ax4.plot(z, spe['TM_s_upper'], label='Fully radiative upper outgoing')

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax3.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax4.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 4)
    ax3.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax3.set_ylabel('$\Gamma /\Gamma_0$')
    ax3.set_xlabel('z/$\lambda$')
    ax4.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', fontsize='small')
    ax2.legend(title='Horizontal Dipoles', fontsize='small')
    ax3.legend(title='Vertical Dipoles', fontsize='small')
    ax4.legend(title='Vertical Dipoles', fontsize='small')
    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/creatore_fig3')
    plt.show()


# Update plot parameters for publication
def update():
    # Set figure size
    WIDTH = 412.56  # the number (in pt) latex spits out when typing: \the\linewidth (paper 246, thesis 412.56)
    FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list

    # Update rcParams for figure size
    params = {
        'font.size': 11.0,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'savefig.dpi': 900,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'figure.figsize': fig_dims,
    }
    plt.rcParams.update(params)


if __name__ == "__main__":
    SAVE = True

    update()
    # supplementary_material1()
    supplementary_material2()

    # fig7()
    # further_work1()
