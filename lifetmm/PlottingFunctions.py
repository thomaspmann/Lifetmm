def plot_leaky():
    # ------- Plots -------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, res['leaky']['avg'])
    if st.supports_guiding():
        ax2.plot(z, res['guided']['avg'])

    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z (nm)')
    ax1.set_title('Leaky')
    ax2.set_title('Guided')

    for zb in st.get_layer_boundaries()[:-1]:
        ax1.axvline(x=zb, color='k', lw=2)
        ax2.axvline(x=zb, color='k', lw=2)

    # Draw rectangles for the refractive index
    ax1b = ax1.twinx()
    ax2b = ax2.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
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
        plt.savefig('../Images/individual')


def plot_leaky_guided():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, res['leaky']['avg'])
    if st.supports_guiding():
        ax2.plot(z, res['guided']['avg'])

    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z (nm)')
    ax1.set_title('Leaky')
    ax2.set_title('Guided')

    for zb in st.get_layer_boundaries()[:-1]:
        ax1.axvline(x=zb, color='k', lw=2)
        ax2.axvline(x=zb, color='k', lw=2)

    # Draw rectangles for the refractive index
    ax1b = ax1.twinx()
    ax2b = ax2.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
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
        plt.savefig('../Images/individual')

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
        plt.savefig('../Images/total')
    plt.show()
