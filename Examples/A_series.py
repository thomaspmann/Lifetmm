"""
Thin film calculations for A-series.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from lifetmm.Materials import n_dict
from lifetmm.SPE import SPE


def plot_leaky_guided_total():
    """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    # Create Structure
    st = SPE()
    st.add_layer(d_clad * lam0, n_dict['Si'])
    st.add_layer(10, n_dict['TZN'])
    st.add_layer(d_clad * lam0, n_dict['Air'])

    st.set_vacuum_wavelength(lam0)
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
    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax2.set_ylim(0, ax2.get_ylim()[1])
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
    ax1b.set_ylim(ax1.get_ylim())
    ax2b.set_ylim(ax2.get_ylim())
    if SAVE:
        plt.savefig('../Images/{}_individual'.format('chip'))

    fig, ax1 = plt.subplots()
    if st.supports_guiding():
        ax1.plot(z, res['leaky']['avg'] + res['guided']['avg'], label='Avg')
    else:
        ax1.plot(z, res['leaky']['avg'], label='Avg')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    # ax1.legend()
    ax1.set_ylim(0, ax1.get_ylim()[1])
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
    ax2.set_ylim(ax1.get_ylim())
    if SAVE:
        plt.savefig('../Images/10nm')
    plt.show()


def plot_vertical_horizontal_total(sample):
    """Plot SE rates for vertical and horizontal dipoles. Then plot sum for randomly orientated dipole."""
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


def plot_te_tm(sample):
    # Load Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Structure 1
    st = SPE()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(chip['d'], chip['n'])
    st.add_layer(d_clad * lam0, n_dict['Si'])
    st.info()

    result1 = st.calc_spe_structure(th_pow=11)
    try:
        spe1 = result1['leaky']['avg'] + result1['guided']['avg']
    except KeyError:
        spe1 = result1['leaky']['avg']

    z = result1['z']
    z = st.calc_z_to_lambda(z)

    # ------- Plots -------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result1['leaky']['avg'])
    try:
        ax2.plot(z, result1['guided']['avg'])
    except KeyError:
        pass

    # Labels
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
    ax1.plot(z, spe1)

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    # ax1.legend()

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
        plt.savefig('../Images/{}_total_te_tm'.format(chip['Sample ID']))
    plt.show()


def fig6(sample):
    # Load Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Structure 1
    st = SPE()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(chip['d'], chip['n'])
    st.add_layer(d_clad * lam0, n_dict['Air'])
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s'], label='TM')

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=2)
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=2)

    # ax1.set_ylim(0, 1.4)
    # ax2.set_ylim(0, 1.4)
    ax1.set_title('{}: Spontaneous Emission Rate.'.format(chip['Sample ID']))
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    if SAVE:
        plt.savefig('../Images/{}_fig6'.format(chip['Sample ID']))
    plt.show()


def fp():
    # Structure 1
    st1 = SPE()
    st1.add_layer(d_clad * lam0, n_dict['SiO2'])
    st1.add_layer(30, n_dict['TZN'])
    st1.add_layer(d_clad * lam0, n_dict['Air'])
    st1.set_vacuum_wavelength(lam0)
    st1.info()

    # Structure 2
    st2 = SPE()
    st2.add_layer(d_clad * lam0, n_dict['SiO2'])
    st2.add_layer(500, n_dict['TZN'])
    st2.add_layer(d_clad * lam0, n_dict['Air'])
    st2.set_vacuum_wavelength(lam0)
    st2.info()

    from lifetmm.SPE import purcell_factor
    purcell_factor(st1, st2, layer=1)


def vary_film_thickness():
    """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    d_list = np.arange(5, 4000, 30, dtype=int)
    # d_list = np.geomspace(1, 25000, num=20, dtype=int)
    y = []
    for i, d in enumerate(d_list):

        # Create Structure
        st = SPE()
        st.add_layer(d_clad * lam0, n_dict['SiO2'])
        st.add_layer(d, n_dict['TZN'])
        st.add_layer(d_clad * lam0, n_dict['Air'])

        st.set_vacuum_wavelength(lam0)
        st.info()

        # Calculate
        res = st.calc_spe_structure(th_pow=11)
        z = res['z']
        z = st.calc_z_to_lambda(z)

        fig, ax1 = plt.subplots()
        if st.supports_guiding():
            ax1.plot(z, res['leaky']['avg'] + res['guided']['avg'], label='Avg')
        else:
            ax1.plot(z, res['leaky']['avg'], label='Avg')
        ax1.set_ylabel('$\Gamma / \Gamma_0$')
        ax1.set_xlabel('Position z ($\lambda$)')
        ax1.legend()
        ax1.set_ylim(0, 4.2)  # ax1.get_ylim()[1]
        # Draw rectangles for the refractive index
        ax2 = ax1.twinx()
        for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
            z0 = st.calc_z_to_lambda(z0)
            dz = st.calc_z_to_lambda(dz, center=False)
            rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
            ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
        ax2.set_ylabel('n')
        ax2.set_ylim(ax1.get_ylim())
        ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
        ax1.patch.set_visible(False)  # hide ax1'canvas'

        for zb in st.get_layer_boundaries()[:-1]:
            zb = st.calc_z_to_lambda(zb)
            ax1.axvline(x=zb, color='k', lw=2)
        plt.title('Film thickness: {} nm'.format(d))

        if SAVE:
            plt.savefig('../Images/Vary thickness/{:05d}'.format(i))
        # plt.show()

        y.append(np.average(res['leaky']['avg']))
    y = np.array(y)
    print(y)
    np.savez('results', d_list, y)


# Update plot parameters for publication
def update():
    # Set figure size
    WIDTH = 15  # 246  # the number (in pt) latex spits out when typing: \the\linewidth
    FACTOR = 1.0  # the fraction of the width you'd like the figure to occupy
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0  # / 72.27
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
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'figure.figsize': fig_dims,
    }
    plt.rcParams.update(params)


def plot_structure():
    # Create Structure
    st = SPE()
    st.add_layer(d_clad * lam0, n_dict['Si'])
    st.add_layer(d_clad * lam0 / 2, n_dict['TZN'])
    st.add_layer(d_clad * lam0, n_dict['Air'])

    st.set_vacuum_wavelength(lam0)
    st.info()

    fig, ax = plt.subplots()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax.set_ylabel('n')
    ax.set_ylim([0, 1.2 * max(st.n_list)])
    lb = st.calc_z_to_lambda(st.d_cumulative[0]) - d_clad
    ub = st.calc_z_to_lambda(st.d_cumulative[-1])
    ax.set_xlim([lb, ub])
    # ax.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    # ax.patch.set_visible(False)  # hide ax1'canvas'

    for zb in st.get_layer_boundaries()[:-1]:
        zb = st.calc_z_to_lambda(zb)
        ax.axvline(x=zb, color='k', lw=2)
    plt.title('Film thickness: {} nm'.format(d_clad * lam0 / 2))
    ax.set_xlabel('Position z ($\lambda$)')
    if SAVE:
        plt.savefig('../Images/structure.png')
    plt.show()


if __name__ == "__main__":
    SAVE = True  # Save figs and data? (bool)

    # Set vacuum wavelength
    lam0 = 1550

    # Cladding thickness (in units of lam0)
    d_clad = .1

    update()

    plot_leaky_guided_total()
    # plot_vertical_horizontal_total(sample='T2')
    # fp()
    # plot_te_tm(sample='T2')
    # fig6(sample='T2')
    # plot_structure()
    # vary_film_thickness()
