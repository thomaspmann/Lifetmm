"""
Thin film calculations for T-series. Purcell factor.
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm


def purcell_factor(chip, n1, n2):
    # Structure 1
    st1 = LifetimeTmm()
    st1.set_vacuum_wavelength(lam0)
    st1.add_layer(d_clad * lam0, n['SiO2'])
    st1.add_layer(chip['d'], chip['n'])
    st1.add_layer(d_clad * lam0, n[n1])
    st1.info()

    # Structure 2
    st2 = LifetimeTmm()
    st2.set_vacuum_wavelength(lam0)
    st2.add_layer(d_clad * lam0, n['SiO2'])
    st2.add_layer(chip['d'], chip['n'])
    st2.add_layer(d_clad * lam0, n[n2])
    st2.info()

    # ------- Calculations -------
    # Calculate spontaneous emission for leaky and guided modes
    result1 = st1.calc_spe_structure(th_pow=11)
    try:
        spe1 = result1['leaky']['avg'] + result1['guided']['avg']
    except KeyError:
        spe1 = result1['leaky']['avg']

    result2 = st2.calc_spe_structure(th_pow=11)
    try:
        spe2 = result2['leaky']['avg'] + result2['guided']['avg']
    except KeyError:
        spe2 = result2['leaky']['avg']

    z = result1['z']
    z = st1.calc_z_to_lambda(z)
    boundaries = [st1.calc_z_to_lambda(i) for i in st1.get_layer_boundaries()[:-1]]

    fp = np.mean(spe2) / np.mean(spe1)
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
    # ax1.set_ylim(0.7, 2)
    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$)')
    ax1.legend(title='Leaky')
    ax2.legend(title='Guided')

    if SAVE:
        plt.savefig('../Images/purcell_factor_indivd_' + chip['name'])

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
    if SAVE:
        plt.savefig('../Images/purcell_factor_total_' + chip['name'])
    plt.show()


if __name__ == "__main__":
    SAVE = False  # Save figs and data? (bool)

    # Set vacuum wavelength
    lam0 = 1535

    # Cladding thickness (in units of lam0)
    d_clad = 0  # 1.6

    n = {'Air': 1,
         'Water': 1.3183,
         'SiO2': 1.442,
         'Glycerol': 1.46,
         'EDTS': 1.56,
         'Cassia Oil': 1.6,
         'Diiodomethane': 1.71
         }

    # Chip parameters
    t12 = {'name': 'T12', 'n': 1.4914, 'd': 566}
    t13 = {'name': 'T13', 'n': 1.6235, 'd': 338}

    purcell_factor(chip=t13, n1='Air', n2='Cassia Oil')
