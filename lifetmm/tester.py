import numpy as np
import matplotlib.pyplot as plt


def g(r_bot, r_top):
    num = (1+np.sqrt(r_bot))**2 * (1-r_top)
    denom = (1-np.sqrt(r_top*r_bot))**2
    return num / denom


def contourPlot():
    plt.figure()

    x = np.linspace(0, 0.9, 100)     # r
    y = np.linspace(0, 0.9, 100)   # n20
    X, Y = np.meshgrid(x, y)

    zs = np.array([g(r_bot, r_top)
                   for r_bot, r_top in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)

    origin = 'lower'
    # origin = 'upper'
    lv = 20  # Levels of colours
    CS = plt.contourf(X, Y, Z, lv,
                      # levels=np.arange(0, 100, 5),
                      cmap=plt.cm.plasma,
                      origin=origin)

    CS2 = plt.contour(CS,
                      levels=CS.levels[::4],
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

    # plt.title()
    # plt.savefig('Images/contourfplot.png', dpi=900)

    plt.show()


if __name__ == "__main__":

    contourPlot()