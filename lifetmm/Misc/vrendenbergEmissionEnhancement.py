import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def finesse(r_back, r_out):
    den = 1 - np.sqrt(r_back*r_out)
    num = np.pi * (r_back*r_out)**(1/4)
    return num / den


def Q_cavity(lam, L_cav, r_back, r_out):
    F = finesse(r_back, r_out)
    return (2*L_cav*F)/lam


def Q_experimental(lam, lam_fwhm):
    return lam / lam_fwhm


def g_e(r_back, r_out, eps=2, tau_0=10, tau_cav=10):
    """ Rate enhancement factor G for the peak emission rate on the optical axis through the top reflector
    of excited ions in the resonance antinode position (mid plane of the active region.

    Valid for all combinations of r_bot and r_top.

    Energy conservation requirements imply that g should be multiplied by the ratio of the
    excited state lifetimes with and without the cavity tau_cav/tau_0 to obtain the true
    intensity enhancement factor.

    eps is the antinode enhancement factor, which has a valie of 2 for atoms within an electric
    field antinode of the emitting optical mode, and 0 for atoms exactly within a node.
    """
    num = (1+np.sqrt(r_back))**2 * (1-r_out)
    den = (1-np.sqrt(r_out*r_back))**2
    return (tau_cav/tau_0) * (eps/2) * (num / den)


def vrendenberg_pl_enhancement():
    """ Evaluate the numbers as given in pg716, confined electrons
    and photons - New Physics and Applications, Vrendenberg, 1995
    """
    lam = 1550
    r_back = 0.997
    r_out = 0.985
    L_cav = 0.85 * lam

    # Theoretical Q
    print(Q_cavity(lam, L_cav, r_back, r_out))
    # Experimental Q
    print(Q_experimental(1540, 5))
    # Emission Rate Intensity Enhancement
    print(g_e(r_back, r_out))


def contourPlot():
    plt.figure()

    x = np.linspace(0, 0.9, 200)    # r_bot
    y = np.linspace(0, 0.9, 200)    # r_top
    X, Y = np.meshgrid(x, y)

    zs = np.array([g_e(r_bot, r_top)
                   for r_bot, r_top in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)

    origin = 'lower'
    # origin = 'upper'
    lv = 30  # Levels of colours
    CS = plt.contourf(X, Y, Z, lv,
                      # levels=np.arange(0, 100, 5),
                      # norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                      cmap=plt.cm.plasma,
                      origin=origin)

    CS2 = plt.contour(CS,
                      levels=CS.levels[::3],
                      colors='k',
                      origin=origin,
                      hold='on')

    plt.xlabel('$r_{bot}$')
    plt.ylabel('$r_{top}$')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Enhancement of radiative decay')
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)

    # plt.title('')
    plt.savefig('Images/Vrendenberg_contourfplot.png', dpi=900)

    plt.show()


def varyBot():
    r_top = 0.04
    r_bot = np.linspace(0.1, 0.9, 200)

    plt.figure()
    plt.plot(r_bot, g_e(r_bot, r_top))
    plt.xlabel('Reflectivity of the medium')
    plt.ylabel('Emission rate enhancement')
    plt.show()

if __name__ == "__main__":
    vrendenberg_pl_enhancement()
    # contourPlot()
    # varyBot()
