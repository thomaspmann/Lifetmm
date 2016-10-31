"""
Thin film calculations.
"""

import matplotlib.pyplot as plt
import numpy as np
from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm
SAVE = True  # Save figs and data? (bool)


def t2_spe_vs_z():
    # Vacuum wavelength
    lam0 = 1550

    # Create plot
    f, ax = plt.subplots(figsize=(15, 7))

    n_list = [1.2, 1.5, 2]
    for n in n_list:
        print('Evaluating n={:g}'.format(n))
        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        st.add_layer(4000, 1.45)  # 1.45 is silica glass substrate
        st.add_layer(980, 1.56)
        st.add_layer(4000, n)

        # Calculate spontaneous emission over whole structure
        result = st.calc_spe_structure_radiative()
        z = result['z']
        spe = result['spe']['total']
        # Plot
        ax.plot(z, spe, label=n, lw=2)
        ax.axhline(y=n, xmin=0.8, xmax=1, ls='dotted', color='k', lw=2)

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax.axvline(z, color='k', lw=2)

    ax.axhline(1.56, ls='--', color='k', lw=2)
    ax.set_title('Spontaneous emission rate at boundary for semi-infinite media. LHS n=1.57.')
    ax.set_ylabel('$\Gamma / \Gamma_0$')
    ax.set_xlabel('Position z ($\lambda$/2$\pi$)')
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig('../Images/spe_vs_z_t2_sub.png', dpi=300)
    plt.show()


def t2_spe_vs_n():
    # Vacuum wavelength
    lam0 = 1550

    # n_list = np.linspace(1, 1.47, 20)
    n_list = [1, 1.33, 1.37, 1.47]
    spe_list = []
    for n in n_list:
        print('Evaluating n={:g}'.format(n))
        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        # st.add_layer(0, 1.45)
        st.add_layer(0, 1.56)
        st.add_layer(980, 1.56)
        st.add_layer(0, n)
        # Calculate spontaneous emission of layer 0 (1st)
        result = st.spe_layer_radiative(layer=1, emission='Lower', th_num=13)
        spe = result['spe']['total']
        result = st.spe_layer_radiative(layer=1, emission='Upper', th_num=13)
        spe += result['spe']['total']
        # Take average
        spe /= 2
        spe = np.mean(spe)
        # Normalise to bulk (so that spe = 1 in the doped layer)
        spe -= 0.5
        # Append to list
        spe_list.append(spe)

    # Convert lists to arrays
    n_list = np.array(n_list)
    spe_list = np.array(spe_list)

    # Plot
    f, ax = plt.subplots(figsize=(15, 7))
    ax.plot(n_list, spe_list, '.-')
    ax.set_title('Decay rate of T2.')
    ax.set_ylabel('$\Gamma / \Gamma_0$')
    ax.set_xlabel('n')
    plt.tight_layout()
    if SAVE:
        plt.savefig('../Images/spe_vs_n_t2_nosubstrate.png', dpi=300)
        np.savez('../Data/spe_vs_n_t2_nosubstrate', n=n_list, spe=spe_list)
    plt.show()


def load_data():
    data = np.load('./lifetmm/Data/spe_vs_n.npz')
    print(data._files)


if __name__ == "__main__":
    t2_spe_vs_z()
    # t2_spe_vs_n()
