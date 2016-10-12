"""
Recreate plots from

    'Measuring and modifying the spontaneous emission rate of erbium near an interface'
    by Snoeks, E, Lagendijk, A, Polman, A
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm
SAVE = True  # Save figs? (bool)


def fig3():
    """ Plot the average decay rate of layer(normalised to bulk n) vs n of semi-infinite half space.
    """
    # Vacuum wavelength
    lam0 = 1550

    n_list = np.linspace(1, 2, 50)
    spe_list = []
    for n in n_list:
        print('Evaluating n={:g}'.format(n))
        # Create structure
        st = LifetimeTmm()
        st.set_wavelength(lam0)
        st.add_layer(1550, 1.5)
        st.add_layer(0, n)
        # Calculate spontaneous emission of layer 0 (1st)
        result = st.spe_layer(layer=0, emission='Lower')
        spe = result['spe']['total']
        result = st.spe_layer(layer=0, emission='Upper')
        spe += result['spe']['total']
        # Take average
        spe /= 2
        # Normalise to bulk
        spe = np.mean(spe) - 1.5
        # Append to list
        spe_list.append(spe)
    spe_list = np.array(spe_list)

    # Plot
    f, ax = plt.subplots(figsize=(15, 7))
    ax.plot(n_list, spe_list)
    ax.set_title('Average spontaneous emission rate over doped layer (d=1550nm) compared to bulk.')
    ax.set_ylabel('$\Gamma / \Gamma_1.5$')
    ax.set_xlabel('n')
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig('../Images/spe_vs_n.png', dpi=300)
        np.savez('../Data/spe_vs_n', n=n_list, spe=spe_list)
    plt.show()


def fig4():
    """ Silicon to air semi-infinite half spaces.
    """
    # Vacuum wavelength
    lam0 = 1550
    # Plotting units
    units = lam0 / (2 * pi)
    # Create structure 1
    st = LifetimeTmm()
    st.set_wavelength(lam0)
    st.add_layer(4*units, 3)
    st.add_layer(4*units, 1.5)
    st.info()
    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    # Shift so centre of structure at z=0
    z -= st.get_structure_thickness() / 2
    spe = result['spe']

    # Plot spontaneous emission rates
    f, ax = plt.subplots(figsize=(15, 7))
    ax.plot(z/units, spe['total'], label='n=3', lw=2)

    # Create structure 2
    st = LifetimeTmm()
    st.set_wavelength(lam0)
    st.add_layer(4*units, 1)
    st.add_layer(4*units, 1.5)
    st.info()
    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    # Shift so centre of structure at z=0
    z -= st.get_structure_thickness() / 2
    spe = result['spe']
    # Plot spontaneous emission rates
    ax.plot(z/units, spe['total'], label='n=1', lw=2)

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        # Shift so centre of structure at z=0
        z -= st.get_structure_thickness() / 2
        ax.axvline(z/units, color='k', lw=2)

    ax.axhline(y=1, xmin=0, xmax=0.4, ls='dotted', color='k', lw=2)
    ax.axhline(y=3, xmin=0, xmax=0.4, ls='dotted', color='k', lw=2)
    ax.axhline(1.5, ls='--', color='k', lw=2)
    ax.set_title('Spontaneous emission rate at boundary for semi-infinite media. RHS n=1.5.')
    ax.set_ylabel('$\Gamma / \Gamma_0$')
    ax.set_xlabel('Position z ($\lambda$/2$\pi$)')
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig('../Images/spe_vs_n.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    fig3()
    # fig4()
