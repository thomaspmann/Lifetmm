import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def g(r_bot, r_top):
    """
    Rate enhancement factor G for the peak emission rate on the optical axis through the top reflector
    of excited ions in the resonance antinode position (mid plane of the active region.

    Valid for all combinations of r_bot and r_top.

    Energy conservation requirements imply that g should be multiplied by the ratio of the
    excited state lifetimes with and without the cavity tau_cav/tau_0 to obtain the true
    intensity enhancement factor.

    :param r_bot: Reflectivity of the bottom side of the cavity
    :param r_top: Reflectivity of the top side of the cavity
    :return:
    """
    num = (1+np.sqrt(r_bot))**2 * (1-r_top)
    den = (1-np.sqrt(r_top*r_bot))**2
    return num / den


def contourPlot():
    plt.figure()

    x = np.linspace(0, 0.99, 200)    # r_bot
    y = np.linspace(0, 0.99, 200)    # r_top
    X, Y = np.meshgrid(x, y)

    zs = np.array([g(r_bot, r_top)
                   for r_bot, r_top in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)

    origin = 'lower'
    # origin = 'upper'
    lv = 30  # Levels of colours
    CS = plt.contourf(X, Y, Z, lv,
                      # levels=np.arange(0, 100, 5),
                      norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
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
    # plt.savefig('Images/contourfplot.png', dpi=900)

    plt.show()


def varyBot():
    r_top = 0.04
    r_bot = np.linspace(0.1, 0.11, 200)

    plt.figure()
    plt.plot(r_bot, g(r_bot, r_top))
    plt.xlabel('Reflectivity of the medium')
    plt.ylabel('Emission rate enhancement')
    plt.show()

if __name__ == "__main__":

    # contourPlot()
    varyBot()
