"""
Example scripts for the LifetimeTmm package.
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.SpontaneousEmissionRate import LifetimeTmm


def example1():
    """ Silicon to air semi-infinite half spaces."""
    # Vacuum wavelength
    lam0 = 1550

    n_list = np.linspace(1, 2, 3)
    spe_list = []
    for n in n_list:
        print('Evaluating n={}'.format(n))
        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        st.add_layer(1550, 1.5)
        st.add_layer(1550, n)
        # Calculate spontaneous emission over whole structure
        result = st.calc_spe_structure_leaky()
        z = result['z']
        spe = result['spe']['total']

        # Only get spe rates in the active layer and then average
        ind = np.where(z <= 1550)
        spe = spe[ind]
        spe = np.mean(spe) - 1.5
        spe_list.append(spe)

    spe_list = np.array(spe_list)
    # Plot spontaneous emission rates vs n
    f, ax = plt.subplots(figsize=(15, 7))
    ax.plot(n_list, spe_list)

    ax.set_title('Average spontaneous emission rate over doped layer (d=1550nm) compared to bulk.')
    ax.set_ylabel('$\Gamma / \Gamma_1.5$')
    ax.set_xlabel('n')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example1()
