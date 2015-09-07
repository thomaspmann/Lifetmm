from __future__ import division, print_function, absolute_import

# from .lifetmm_core import *
from lifetmm import *

from numpy import pi, linspace, inf, array, sum
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# To run a sample use the following in python console:

# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180

mmTOnm = 1E6


def test():
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3, 3]
    n_listB = [1.5, 1.5, 1.5, 1.5]

    data = np.zeros(sum(d_list[1:-1]))

    E_avg = 0
    runs = 0

    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [True, False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                         TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])

    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data, label='data')
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.savefig('figs/test.png')
    plt.show()

def test2():
    """
    Same as test() but no normalising
    """
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3, 3]
    n_listB = [1.5, 1.5, 1.5, 1.5]

    data = np.zeros(sum(d_list[1:-1]))

    E_avg = 0
    runs = 0

    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [True, False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])

    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data)
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.savefig('figs/test2.png')
    plt.show()


def sample1():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = 1550
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3]
    n_listB = [1.5, 1.5, 1]

    data = np.zeros(sum(d_list[1:-1]))

    nRange = linspace(1, 3, num=100)
    ydata = np.zeros(len(nRange))

    for i, n in enumerate(nRange):
        E_avg = 0
        runs = 0
        n_list[2] = n
        for th in th_0:
            for pol in ['s', 'p']:
                for rev in [True, False]:
                    runs += 1
                    # data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                    #          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                    E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                              TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
        ydata[i] = E_avg/runs

    # Normalise
    # data /= runs
    # E_avg /= runs

    # print(ydata)

    plt.figure()
    plt.plot(nRange, ydata)
    plt.xlabel('Refractive Index of Sensing Medium')
    plt.ylabel('Average |E|$^2$Intensity Inside the Erbium Layer')
    plt.title('Average E-Field Intensity in Device')
    plt.savefig('figs/BigOleLoop_air.png')
    plt.show()


def sample2():
    """
    Show the LDOS inside the erbium layer relative to the bulk
        Averaged over all angles (0-90 degrees)
        Averaged over s and p polarisations
    """

    # list of wavelengths to evaluate
    lambda_vac = 1550
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1]
    n_listB = [1.5, 1.5, 1.5]

    data = np.zeros(sum(d_list[1:-1]))

    E_avg = 0
    runs = 0
    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [True, False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                         TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])

    # Normalise
    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')
    plt.savefig('figs/LDOS.png')
    plt.show()


def sample3():
    """
    Show the LDOS inside the erbium layer relative to the bulk
        Averaged over all angles (0-90 degrees)
        Averaged over s and p polarisations
    """

    # list of wavelengths to evaluate
    lambda_vac = 1550
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 1000, 200, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1, 1.41, 1.41]
    n_listB = [1.5, 1.5, 1.5, 1.5, 1.5]

    data = np.zeros(sum(d_list[1:-1]))
    E_avg = 0
    runs = 0

    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [True, False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                         TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
    # Normalise over all runs
    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data)
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.savefig('figs/LDOS_medium_%.4f.png' % E_avg)
    plt.show()

def samplePol():
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_list = [1550]  # in nm
    # incoming light angle
    th_0 = linspace(0, 90, num=90+1, endpoint=False) # in degrees (convert in function argument)

    # ------------- DO CALCULATIONS  -----------------
    # list of layer thicknesses in nm
    d_list = [inf, 10000, 10000, inf]
    # list of refractive indices
    n_list_med = [1, 1.5, 3, 3]
    n_list_bulk = [1, 1.5, 1.5, 1.5]
    data_s = np.zeros(sum(d_list[1:-1]))
    data_p = np.zeros(sum(d_list[1:-1]))
    for lambda_vac in lambda_list:
        for th in th_0:
            a = TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 's')['E_square']
            b = TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 's')['E_square']
            data_s += (a/b)
            c = TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 'p')['E_square']
            d = TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 'p')['E_square']
            data_p += (c/d)
    data_s /= (len(lambda_list)*len(th_0))  # Normalise again - average over loops
    data_p /= (len(lambda_list)*len(th_0))  # Normalise again
    data = (data_s + data_p) / 2  # Take average for unpolarised light
    # ----------------------- END -----------------------------

    plt.figure()
    plt.plot(data)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')
    # plt.savefig('moreColors.png')
    plt.show()

test2()
