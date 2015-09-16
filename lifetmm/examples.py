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


def mcgehee():
    '''
    Copy of the stanford group simulation at 600nm
    '''
    # list of wavelengths to evaluate
    lambda_vac = 600
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252+7.25439j]


    data = np.zeros(sum(d_list[1:-1]))

    E_avg = 0
    runs = 0

    for th in [0]:
        for pol in ['s']:
            for rev in [False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev, glass=True)['E_square'])
                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])

    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data, label='data')
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    # for i, xmat in enumerate(dsum):
        # plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    # plt.savefig('figs/fwdbkwd_sp.png')
    plt.show()

def test():
    """

    """
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, inf]
    # list of refractive indices
    n_list = [3, 3, 1.5, 1.5]
    n_listRef = [3, 1.5, 1.5, 1.5]

    # Initialise and run
    E_profile = np.zeros(sum(d_list[1:-1]))
    E_profileRef = np.zeros(sum(d_list[1:-1]))
    E_avg = 0
    runs = 0
    weighting = 0
    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [False, True]:
                runs += 1
                if rev:
                    weighting = n_list[-1]
                elif not rev:
                    weighting = n_list[0]

                weighting *= (np.sin(th * degree) * np.cos(th * degree))

                E_profile += (weighting * TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'])
                # E_profileRef += (weighting * TransferMatrix(d_list, n_listRef, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (weighting * TransferMatrix(d_list, n_list, lambda_vac,
                                                    th * degree, pol, reverse=rev)['E_avg'][1])

    E_profile /= runs
    # E_profileRef /= runs
    # E_final = np.true_divide(E_profile, E_profileRef)
    # E_final = E_profile/E_profileRef
    E_avg /= runs

    plt.figure()
    plt.plot(E_profile, 'b', label='E')
    # plt.plot(E_profileRef, 'r', label='ref')
    # plt.plot(E_final, 'g', label='Normalised')
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        plt.text(xmat-300, max(E_profile)*0.99,'n: %.2f' % n_list[i+1])
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend(loc='best')
    plt.savefig('figs/test3.png')
    plt.show()


def sample1():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3]
    n_listB = [1.5, 1.5, 1]

    data = np.zeros(sum(d_list[1:-1]))

    nRange = linspace(1, 1.8, num=200)

    ydata = np.zeros(len(nRange))
    for i, n in enumerate(nRange):
        print('i is %d and n is %f' % (i, n))
        E_avg = 0
        runs = 0
        n_list[2] = n
        print('i is %d and n is %f' % (i, n_list[2]))
        for th in th_0:
            for pol in ['s', 'p']:
                for rev in [True, False]:
                    runs += 1
                    # data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                    #          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                    E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
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
    plt.savefig('figs/erlayersandwich2lowres.png')
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


def amerov():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3]
    n_listB = [1.5, 1.5, 1]

    mM = linspace(60, 100, num=200)
    nRange = [2.73E-5 * x + 1.37 for x in mM]
    # nRange = 1.325 + mM * 2.73E-5

    ydata = np.zeros(len(nRange))
    for i, n in enumerate(nRange):
        print('i is %d and n is %f' % (i, n))
        E_avg = 0
        runs = 0
        n_list[2] = n
        print('i is %d and n is %f' % (i, n_list[2]))
        for th in th_0:
            for pol in ['s', 'p']:
                for rev in [True, False]:
                    runs += 1
                    # data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                    #          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                    E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
        ydata[i] = E_avg/runs

    plt.figure()
    plt.plot(mM, 1/ydata)
    plt.xlabel('Glucose concentration in water (mM)')
    plt.ylabel('Lifetime (1/E^2)')
    plt.title('Average E-Field Intensity in Device')
    plt.savefig('figs/amerov1p37_3.png')
    plt.show()


def finger():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = 1537
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, 100, 1E5, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1, 1.37, 1.37]

    mM = linspace(0, 20, num=20)
    nRange = [2.73E-5 * x + 1.37 for x in mM]
    # nRange = 1.325 + mM * 2.73E-5

    ydata = np.zeros(len(nRange))
    for i, n in enumerate(nRange):
        print('i is %d and n is %f' % (i, n))
        E_avg = 0
        runs = 0
        n_list[4] = n
        print('i is %d and n is %f' % (i, n_list[4]))
        for th in th_0:
            for pol in ['s', 'p']:
                for rev in [True, False]:
                    runs += 1
                    # data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                    #          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                    E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
        ydata[i] = E_avg/runs

    plt.figure()
    plt.plot(mM, 1/ydata)
    plt.xlabel('Glucose concentration in water (mM)')
    plt.ylabel('Lifetime (1/E^2)')
    plt.title('Average E-Field Intensity in Device')
    plt.savefig('figs/finger.png')
    plt.show()


test()
