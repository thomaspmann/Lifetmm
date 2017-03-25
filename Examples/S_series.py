"""
Thin film calculations for S-series.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lifetmm.SpontaneousEmissionRate import LifetimeTmm


def plot():
    # Structure 1
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, n_dict['SiO2'])
    st.add_layer(1 * lam0, n_dict['TZN'])
    st.add_layer(100, 1.6)
    st.add_layer(1 * lam0, n_dict['TZN'])
    st.add_layer(d_clad * lam0, n_dict['Air'])
    st.info()

    result1 = st.calc_spe_structure(th_pow=11)
    try:
        spe1 = result1['leaky']['avg'] + result1['guided']['avg']
    except KeyError:
        spe1 = result1['leaky']['avg']

    z = result1['z']
    z = st.calc_z_to_lambda(z)
    boundaries = [st.calc_z_to_lambda(i) for i in st.get_layer_boundaries()[:-1]]

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

    for i in boundaries:
        ax1.axvline(x=i, color='k', lw=2)
        ax2.axvline(x=i, color='k', lw=2)

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
        plt.savefig('../Images/{}_individual'.format('s-series'))

    fig, ax1 = plt.subplots()
    ax1.plot(z, spe1)

    # Plot internal layer boundaries
    for i in boundaries:
        ax1.axvline(i, color='k', lw=2)

    # Labels
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    # ax1.legend()

    # Draw rectangles for the refractive index
    ax1b = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.2)
        ax1b.add_patch(rect)
        ax1b.set_ylabel('n')
        ax1b.set_ylim(0, 1.5 * max(st.n_list.real))
    ax1.set_zorder(ax1b.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    if SAVE:
        plt.savefig('../Images/{}_total'.format('s-series'))
    plt.show()


if __name__ == "__main__":
    SAVE = True  # Save figs and data? (bool)

    # Set vacuum wavelength
    lam0 = 1540

    # Cladding thickness (in units of lam0)
    d_clad = 1.5

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

    # excitation_profile()
    plot()
