"""
Thin film calculations for T-series. Purcell factor.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lifetmm.SpontaneousEmissionRate import LifetimeTmm


def plot(sample):
    # Load Data
    df = pd.read_csv('../Data/Screening.csv', index_col='Sample ID')
    n = df.loc[sample]['n']
    d = df.loc[sample]['d'] * 1000  # in nm not um
    chip = {'Sample ID': sample, 'n': n, 'd': d}

    # Structure 1
    st1 = LifetimeTmm()
    st1.set_vacuum_wavelength(lam0)
    st1.add_layer(d_clad * lam0, n_dict['SiO2'])
    st1.add_layer(chip['d'], chip['n'])
    st1.add_layer(d_clad * lam0, n_dict['Air'])
    st1.info()

    result1 = st1.calc_spe_structure(th_pow=11)
    try:
        spe1 = result1['leaky']['avg'] + result1['guided']['avg']
    except KeyError:
        spe1 = result1['leaky']['avg']

    z = result1['z']
    z = st1.calc_z_to_lambda(z)
    boundaries = [st1.calc_z_to_lambda(i) for i in st1.get_layer_boundaries()[:-1]]

    # ------- Plots -------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result1['leaky']['avg'])
    try:
        ax2.plot(z, result1['guided']['avg'])
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
    ax1.set_title('Leaky')
    ax2.set_title('Guided')
    if SAVE:
        plt.savefig('../Images/{}_individual.pdf'.format(chip['Sample ID']))

    fig, ax1 = plt.subplots()
    ax1.plot(z, spe1)

    # Plot internal layer boundaries
    for i in boundaries:
        ax1.axvline(i, color='k', ls='--')

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    # ax1.legend()
    if SAVE:
        plt.savefig('../Images/{}_total.pdf'.format(chip['Sample ID']))
    plt.show()


def purcell_factor(chip, n1, n2, layer):
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
    lam0 = 1535

    # Cladding thickness (in units of lam0)
    d_clad = 1.5

    # Dictionary of material refractive indexes
    n_dict = {'Air': 1,
              'Water': 1.3183,
              'SiO2': 1.442,
              'Glycerol': 1.46,
              'EDTS': 1.56,
              'Cassia Oil': 1.6,
              'Diiodomethane': 1.71
              }

    plot(sample='T21')
    # loop_csv()
    # loop_list()
