"""
Thin film calculations for T-series. Purcell factor.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from lifetmm.SpontaneousEmissionRate import LifetimeTmm
from lifetmm.TransferMatrix import TransferMatrix


def excitation_profile(sample):
    """Plot the E field in the layer of the angled excitation of the laser @ 976nm."""
    # Load Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # convert nm to um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    st = TransferMatrix()
    st.add_layer(0, n_dict['Air'])
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(chip['d'], chip['n'])
    st.add_layer(d_clad * lam0, n_dict['Air'])

    # Laser Excitation Wavelength
    st.set_vacuum_wavelength(976)
    st.set_field('E')

    # Angle of incidence (degrees)
    aoi = 60  # In air - onto chip
    from lifetmm.HelperFunctions import snell
    aoi_soi2 = snell(1, n_dict['SiO2'], aoi * (np.pi / 180))  # Set to aoi in silica (radians)
    st.set_incident_angle(aoi_soi2, units='radians')
    st.info()

    # Do calculations
    st.set_polarization('TE')
    result = st.calc_field_structure()
    z = result['z']
    # z = st.calc_z_to_lambda(z)

    y_s = result['field_squared']

    st.set_polarization('TM')
    result = st.calc_field_structure()
    y_p = result['field_squared']

    # Plot results
    fig, ax1 = plt.subplots()
    ax1.plot(z, y_s, label='TE')
    ax1.plot(z, y_p, label='TM')
    ax1.set_xlabel('z (nm)')
    ax1.set_ylabel('Normalized |E|$^2$ Intensity ($|E(z)/E_0(0)|^2$)')
    ax1.set_title('Angle of incidence {}°'.format(aoi))
    ax1.legend(title='Polarization')

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
    if SAVE:
        plt.savefig('../Images/{}_excitation_profile'.format(chip['Sample ID']), dpi=300)
    plt.show()


def plot_leaky_guided_total(sample):
    """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""
    # Load Sample Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Create Structure
    st = LifetimeTmm()
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


def plot_vertical_horizontal_total(sample):
    """Plot SE rates for vertical and horizontal dipoles. Then plot sum for randomly orientated dipole."""
    # Load Sample Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Create Structure
    st = LifetimeTmm()
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
    st = LifetimeTmm()
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
    st = LifetimeTmm()
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


def purcell_factor(sample, n1, n2, layer):
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1e3  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Structure 1
    st1 = LifetimeTmm()
    st1.set_vacuum_wavelength(lam0)
    st1.add_layer(d_clad * lam0, n_dict['SiO2'])
    st1.add_layer(chip['d'], chip['n'])
    st1.add_layer(d_clad * lam0, n_dict[n1])
    st1.info()

    result1 = st1.calc_spe_structure(th_pow=11)
    try:
        spe1 = result1['leaky']['avg'] + result1['guided']['avg']
    except KeyError:
        spe1 = result1['leaky']['avg']
    ind = st1.get_layer_indices(layer)
    fp1 = np.mean(spe1[ind])

    # Structure 2
    st2 = LifetimeTmm()
    st2.set_vacuum_wavelength(lam0)
    st2.add_layer(d_clad * lam0, n_dict['SiO2'])
    st2.add_layer(chip['d'], chip['n'])
    st2.add_layer(d_clad * lam0, n_dict[n2])
    st2.info()

    result2 = st2.calc_spe_structure(th_pow=11)
    try:
        spe2 = result2['leaky']['avg'] + result2['guided']['avg']
    except KeyError:
        spe2 = result2['leaky']['avg']
    ind = st1.get_layer_indices(layer)
    fp2 = np.mean(spe2[ind])

    z = result1['z']
    z = st1.calc_z_to_lambda(z)
    boundaries = [st1.calc_z_to_lambda(i) for i in st1.get_layer_boundaries()[:-1]]

    fp = fp2 / fp1
    print('Purcell Factor: {:e}'.format(fp))

    # ------- Plots -------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result1['leaky']['avg'], label=n1)
    ax1.plot(z, result2['leaky']['avg'], ls='--', label=n2)

    try:
        ax2.plot(z, result1['guided']['avg'], label=n1)
    except KeyError:
        pass

    try:
        ax2.plot(z, result2['guided']['avg'], ls='--', label=n2)
    except KeyError:
        pass

    # Plot internal layer boundaries
    for i in boundaries:
        ax1.axvline(i, color='k', ls='--')
        ax2.axvline(i, color='k', ls='--')

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$)')
    ax1.legend(title='Leaky')
    ax2.legend(title='Guided')
    ax1.set_title('{0}: Purcell Factor: {1:.3f}'.format(chip['Sample ID'], fp))
    if SAVE:
        plt.savefig('../Images/{}_purcell_factor_individ'.format(chip['Sample ID']))

    fig, ax1 = plt.subplots()
    ax1.plot(z, spe1, label=n1)
    ax1.plot(z, spe2, ls='--', label=n2)

    # Plot internal layer boundaries
    for i in boundaries:
        ax1.axvline(i, color='k', ls='--')

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    ax1.legend()
    plt.title('{0}: Purcell Factor: {1:.3f}'.format(chip['Sample ID'], fp))
    if SAVE:
        plt.savefig('../Images/{}_purcell_factor_total'.format(chip['Sample ID']))
    # plt.show()

    return fp


def loop_list():
    # Chip parameters
    t12 = {'Sample ID': 'T12', 'n': 1.4914, 'd': 566}
    t13 = {'Sample ID': 'T13', 'n': 1.6235, 'd': 338}

    # Jaya's chips
    j125 = {'Sample ID': '0p125', 'n': 1.5754, 'd': 1250}
    j500 = {'Sample ID': '0p5', 'n': 1.5850, 'd': 1000}
    j750 = {'Sample ID': '0p75', 'n': 1.6438, 'd': 520}
    j1000 = {'Sample ID': '1', 'n': 1.6682, 'd': 480}

    for chip in [t12, t13]:
        print(chip['Sample ID'])
        purcell_factor(chip=chip, n1='Air', n2='Cassia Oil', layer=1)


def loop_csv():
    # Load Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    # loop through all samples
    samples = df.index.values.tolist()
    fp_dict = dict.fromkeys(samples)
    for sample in samples:
        print('Evaluating {}'.format(sample))
        n = df.loc[sample]['n']
        d = df.loc[sample]['d'] * 1000  # in nm not um
        chip = {'Sample ID': sample, 'n': n, 'd': d}

        # do purcell calculations
        try:
            fp = purcell_factor(chip=chip, n1='Air', n2='Cassia Oil', layer=1)
        except:
            print('Error calculating sample {}'.format(sample))
            fp = np.nan
        fp_dict[sample] = fp

    # Save results to csv (first convert to pandas series)
    s = pd.Series(fp_dict, name='Purcell Factor')
    s.index.name = 'Sample ID'
    s.reset_index()
    s.to_csv('../Data/fp_nWG.csv', header=True)

if __name__ == "__main__":
    SAVE = True  # Save figs and data? (bool)

    # Set vacuum wavelength
    lam0 = 1550

    # Cladding thickness (in units of lam0)
    d_clad = 0.5

    # Dictionary of material refractive indexes
    n_dict = {'Air': 1,
              'Water': 1.3183,
              'SiO2': 1.442,
              'Glycerol': 1.46,
              'EDTS': 1.56,
              'Cassia Oil': 1.6,
              'Diiodomethane': 1.71,
              'Si': 3.4757
              }

    # excitation_profile(sample='T2')
    # plot_leaky_guided_total(sample='T2')
    plot_vertical_horizontal_total(sample='T2')
    # purcell_factor(sample='T2', n1='Air', n2='Si', layer=1)
    # plot_te_tm(sample='T2')
    # fig6(sample='T2')
    # loop_csv()
    # loop_list()
