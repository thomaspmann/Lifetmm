"""
Thin film calculations.
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm


def t2(medium):
    """
    T2 EDTS layer next to medium.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(d_clad * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(d_clad * lam0, medium)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure(th_pow=9)
    z = result['z']
    z = st.calc_z_to_lambda(z)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result['leaky']['avg'], label='leaky')
    try:
        ax2.plot(z, result['guided']['avg'], label='guided')
    except KeyError:
        pass

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(z, color='k', lw=1, ls='--')
        ax2.axvline(z, color='k', lw=1, ls='--')
    # ax1.set_title('Spontaneous emission rate at boundary for semi-infinite media. LHS n=1.57.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$/2$\pi$)')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/t2.png', dpi=300)
    plt.show()


def t2_leaky():
    """
    T2 EDTS layer next to air.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure_leaky(th_pow=9)
    z = result['z']
    z = st.calc_z_to_lambda(z)
    spe = result['spe']

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

    # ax1.set_ylim(0, 4)
    # ax3.set_ylim(0, 6)
    # ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
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
        plt.savefig('../Images/t2_leaky.png', dpi=300)
    plt.show()


def t2_fig4():
    """
    Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky(th_pow=9)
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(15, 5))
    ax1.plot(z, (spe['TM_p_lower'] + spe['TE_lower']) / (spe['TE'] + spe['TM_p']), label='Lower')
    ax1.plot(z, (spe['TM_p_upper'] + spe['TE_upper']) / (spe['TE'] + spe['TM_p']), label='Upper')

    ax2.plot(z, (spe['TM_s_lower']) / spe['TM_s'], label='Lower')
    ax2.plot(z, (spe['TM_s_upper']) / spe['TM_s'], label='Upper')

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    # ax1.set_ylim(0, 1.1)

    # ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('z/$\lambda$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/t2_fig4.png', dpi=300)
    plt.show()


def t2_guided():
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    result = st.calc_spe_structure_guided()
    z = result['z']
    spe = result['spe']
    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')

    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], label='TE + TM')
    ax2.plot(z, spe['TM_s'], label='TM')
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(x=z, color='k', lw=2, zorder=-1)
        ax2.axvline(x=z, color='k', lw=2, zorder=-1)
    # ax1.set_ylim(0, 4)
    # ax2.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. Core n=3.48, Cladding n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    size = 12
    ax1.legend(title='Horizontal Dipoles', prop={'size': size})
    ax2.legend(title='Vertical Dipoles', prop={'size': size})

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/t2_guided.png', dpi=300)
    plt.show()


def t2_spe_vs_n():
    # n_list = np.append(np.linspace(1, 1.45, num=25), np.linspace(1.45, 1.55, num=50))
    # n_list = np.append(n_list, np.linspace(1.55, 2, num=25))
    n_list = [air, water, glycerol]
    spe_list = []
    leaky_list = []
    guided_list = []
    for n in n_list:
        print('Evaluating n={:g}'.format(n))

        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        st.add_layer(0, sio2)
        st.add_layer(d_etds, edts)
        st.add_layer(0, n)
        st.info()

        # Calculate spontaneous emission of layer 0 (1st)
        result = st.calc_spe_structure(th_pow=12)
        leaky = result['leaky']['avg']
        try:
            guided = result['guided']['avg']
        except KeyError:
            guided = 0

        # Average over layer
        leaky = np.mean(leaky)
        guided = np.mean(guided)
        # Append to list
        leaky_list.append(leaky)
        guided_list.append(guided)
        spe_list.append(leaky + guided)

    # Convert lists to arrays
    n_list = np.array(n_list)
    leaky_list = np.array(leaky_list)
    guided_list = np.array(guided_list)
    spe_list = np.array(spe_list)
    print(n_list, spe_list)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='none')
    ax1.plot(n_list, spe_list, '.-', label='leaky + guided')
    ax2.plot(n_list, leaky_list, '.-', label='leaky')
    ax3.plot(n_list, guided_list, '.-', label='guided')
    ax3.set_xlim(1, 2)
    ax1.set_title('Average Spontaneous Emission Rate for Random Orientated Dipole in T2.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('n')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/spe_vs_n.png', dpi=300)
        np.savez('../Data/spe_vs_n', n=n_list, spe=spe_list, guided=guided_list, leaky=leaky_list)
    plt.show()


def load_data():
    data = np.load('./lifetmm/Data/spe_vs_n.npz')
    print(data._files)


def purcell_factor(n1, n2):
    """
    Two structures.
    Leaky and guided separate plots.
    Evaluate purcell factor for randomly orientated dipole averaged over film thickness.
    """

    # Structure 1
    st1 = LifetimeTmm()
    st1.set_vacuum_wavelength(lam0)
    st1.add_layer(d_clad * lam0, sio2)
    st1.add_layer(d_etds, edts)
    st1.add_layer(d_clad * lam0, n[n1])
    st1.info()

    # Structure 2
    st2 = LifetimeTmm()
    st2.set_vacuum_wavelength(lam0)
    st2.add_layer(d_clad * lam0, sio2)
    st2.add_layer(d_etds, edts)
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
        plt.savefig('../Images/T2_purcell_factor')

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
        plt.savefig('../Images/T2_purcell_factor_total')
    plt.show()

if __name__ == "__main__":
    SAVE = True  # Save figs and data? (bool)

    # Journal plotting formatting/saving setup
    import lifetmm.Methods.journalPlotting
    lifetmm.Methods.journalPlotting.update()

    # Set vacuum wavelength
    lam0 = 1535

    # Film thickness
    d_etds = 980
    d_clad = 1.5  # (in units of lam0)

    # Material refractive index at lam0
    sio2 = 1.442
    edts = 1.56
    air = 1
    water = 1.3183
    glycerol = 1.46
    diiodomethane = 1.71

    n = {'Air': 1,
         'Water': 1.3183,
         'SiO2': 1.442,
         'Glycerol': 1.46,
         'EDTS': 1.56,
         'Cassia Oil': 1.6,
         'Diiodomethane': 1.71
         }

    # t2(water)
    # t2_leaky()
    # t2_fig4()
    # t2_guided()
    # t2_spe_vs_n()
    purcell_factor('Air', 'Water')
