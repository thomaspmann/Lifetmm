from __future__ import division, print_function, absolute_import

# from .lifetmm_core import *
from lifetmm import *

from numpy import pi, linspace, inf, array, sum, cos, sin
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


    runs = 0

    for th in [0]:
        for pol in ['s']:
            runs += 1
            data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol)['E_square'])
    data /= runs

    plt.figure()
    plt.plot(data, label='data')
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    # for i, xmat in enumerate(dsum):
        # plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def mcgehee2():
    '''
    Copy of the stanford group simulation at 600nm
    '''

    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252+7.25439j]
    # list of wavelengths to evaluate
    lambda_vac = 600

    structure = LifetimeTmm(d_list, n_list)
    structure.setWavelength(lambda_vac)

    data = np.zeros(sum(d_list[1:-1]))
    for th in [0]:
        for pol in ['s']:
            structure.setPolarization(pol)
            structure.setAngle(th)
            structure.calculate()
            data = getattr(structure, 'E_square')

    plt.figure()
    plt.plot(data, label='data')
    dsum = getattr(structure, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def vrendenberg():
    d_sio2 = 270
    d_si = 115
    n_si = 3.4
    n_sio2 = 1.44
    # list of layer thicknesses in nm
    d_list = [inf, 2000, d_sio2, d_si, d_sio2, d_si, d_sio2,
              540,
              d_sio2,  d_si, d_sio2,  d_si, d_sio2,  d_si, d_sio2,  d_si, 2000, inf]
    # list of refractive indices
    n_list = [1, 1, n_sio2, n_si, n_sio2, n_si, n_sio2,
              1.54,
              n_sio2,  n_si, n_sio2,  n_si, n_sio2,  n_si, n_sio2,  n_si, 1, 1]
    # list of wavelengths to evaluate in nm
    lambda_vac = 1600

    structure = LifetimeTmm(d_list, n_list)
    structure.setWavelength(lambda_vac)

    plt.figure()
    data = np.zeros(sum(d_list[1:-1]))

    theta_max = thetaCritical(m, n_list)
    for th in [0]:
        structure.setAngle(th)
        n_m = n_list[7]
        th_1 = snell(n_m, n_list[0], th)
        th_s = snell(n_m, n_list[-1], th)

        for pol in ['p', 's']:
            structure.setPolarization(pol)

            structure.calculate()
            data = getattr(structure, 'E_square')
            plt.plot(data, label=pol)

    dsum = getattr(structure, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i in dsum:
        plt.axvline(x=i, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def fingdist():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, 10, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1.55, 1, 1.41, 1.41]
    # Initialise plotting data
    dist = []
    data = []
    for i in range(200):  # number of loops
        x = 10 * (i+1)
        dist.append(x)
        d_list[3] = x  # modify the air layer

        E_avg = 0
        runs = 0
        weighting = 0
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (np.sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam,
                                                            th * degree, pol, reverse=rev)['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    plt.figure()
    plt.plot(dist, data, '.-', label='E')

    plt.xlabel('Air gap between 1um n=1.55 spacer placed on the erbium layer and finger (nm)')
    plt.ylabel('Average Normalized |E|$^2$Intensity over Er layer')
    plt.title('Effect of air gap thickness on the avg E in the erbium layer (n_finger = 1.41)', y=1.08)
    plt.savefig('figs/fingdist5.png')
    plt.show()


def samplePol():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3, 3]

    # List of medium refractive indices
    nmed_list = linspace(1, 2, num=500)

    data = []
    for nmed in nmed_list:
        runs = 0
        E_avg = 0
        weighting = 0
        n_list[2] = nmed
        print('Medium n:' + str(nmed))
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (1-np.sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam, th * degree, pol, reverse=rev)
                                                                                                        ['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    data = np.asarray(data)

    plt.figure()
    plt.plot(nmed_list, data, '.-')
    plt.xlabel('n of medium')
    plt.ylabel('Average |E|$^2$Intensity in Erbium layer')
    plt.title('1000nm erbium, 1000nm sensing medium (n = x axis).')
    plt.savefig('figs/samplePol_1min_sin.png')
    plt.show()


def sample():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 100, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1, 1.41]

    # List of medium refractive indices
    nmed_list = linspace(1.4, 1.42, num=500)

    data = []
    for nmed in nmed_list:
        runs = 0
        E_avg = 0
        weighting = 0
        n_list[-1] = nmed
        print('Medium n:' + str(nmed))
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam, th * degree, pol, reverse=rev)
                                                                                                        ['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    data = np.asarray(data)

    plt.figure()
    plt.plot(nmed_list, data, '.-')
    plt.xlabel('n of medium')
    plt.ylabel('Average |E|$^2$Intensity in Erbium layer')
    plt.title('1000nm erbium, 100nm air, 1000nm sensing medium (n = x axis).')
    plt.savefig('figs/sample2.png')
    plt.show()


def finger():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = [1540]
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

if __name__ == "__main__":
    # mcgehee2()
    # test()
    vrendenberg()
    # samplePol()
    # fingdist()
    # sample()
