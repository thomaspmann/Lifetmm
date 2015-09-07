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
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = 1550
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3]
    n_listB = [1.5, 1.5, 1.5]

    data = np.zeros(sum(d_list[1:-1]))
    E_avg =0

    runs = 0

    for th in th_0:
        for pol in ['s', 'p']:
            for rev in [True, False]:
                runs += 1
                data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                         TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1] /
                         TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])

        # E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, 's', reverse=True)['E_avg'] +
        #     TransferMatrix(d_list, n_list, lambda_vac, th * degree, 's', reverse=False)['E_avg']) /2

    # Normalise
    data /= runs
    E_avg /= runs

    print(E_avg)

    plt.figure()
    plt.plot(data)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')
    plt.show()


def sample1():
    """
    Use transfer matrix function to plot structure.
    """
    # --------------- STRUCTURE IN MEDIUM ---------------------
    # list of layer thicknesses in nm
    d_list = [0, 1000, 1000]
    # list of refractive indices
    n_list = [1, 1.5, 3]
    # list of wavelengths
    lambda_vac = 1550  # in nm
    # incoming light angle
    th_0 = 0
    # polarization of incoming light. 's', 'p' or 'u'
    pol = 's'
    data = TransferMatrix(d_list, n_list, lambda_vac, th_0, pol)
    # ----------------------- END -----------------------------

    # --------------- STRUCTURE IN BULK ---------------------
    # list of layer thicknesses in nm
    d_list = [0, 1000, 1000]
    # list of refractive indices
    n_list = [1, 1.5, 1]

    data_bulk = TransferMatrix(d_list, n_list, lambda_vac, th_0, pol)
    # ----------------------- END -----------------------------

    # ---------------------- DO PLOTS --------------------------
    plt.figure()
    plt.plot(data['x_pos'], data['E_square'], 'blue',
             data_bulk['x_pos'], data_bulk['E_square'], 'red')
    plt.legend(['Medium', 'Bulk'])
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')
    # ----------------------- END -----------------------------


def sample2():
    """
    sum E field over incident theta and wavelengths

    Structure: Glass->Erbium->Refractive Index
    Compare to Refractive Index -> Erb -> Glass (reversed)

    Polman uses latter and just 2 layers Ref. Index->glass
    then sums over locations of Er in glass.
    """

    # ---------------- SET UP DEVICE STRUCTURE ----------------
    # list of layer thicknesses in nm
    d_list = [0, 1000, 1000, 0]
    # list of refractive indices
    n_list = [1, 1.5, 3, 1]
    # Initialise structure of device
    sample_T1 = LifetimeTmm(d_list, n_list)
    # ----------------------- END -----------------------------

    # --------------- SET UP SIMULATION PARAMS ----------------
    # list of wavelengths to evaluate
    lambda_list = [1550]  # in nm
    # list of angles to plot
    th_0 = linspace(0, 90, num=90+1)
    # ----------------------- END -----------------------------

    # ------------------ DO CALCULATIONS ----------------------
    # Initialise numpy array to store y data
    data = np.array([0.]*len(sample_T1.x_pos))
    for lambda_vac in lambda_list:
        for th in th_0:
            data += sample_T1(lambda_vac, th)['E_square']
    data /= (len(lambda_list)*len(th_0))  # Normalise again
    x_pos = sample_T1.x_pos
    # ----------------------- END -----------------------------

    # --------------------- DO REVERSE ------------------------
    data_rev = np.array([0.]*len(sample_T1.x_pos))
    for lambda_vac in lambda_list:
        for th in th_0:
            data_rev += sample_T1.reverse(lambda_vac, th)['E_square']
    data_rev /= (len(lambda_list)*len(th_0))  # Normalise again
    # ----------------------- END -----------------------------

    # ---------------------- DO PLOTS --------------------------
    plt.figure()
    plt.plot(x_pos, data, 'blue', x_pos, data_rev, 'red')
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')
    plt.legend(['Forward', 'Backward'])
    # ----------------------- END -----------------------------


def sample3():
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_list = [1550]  # in nm
    # incoming light angle
    th_0 = linspace(0, 90, num=45, endpoint=False)  # in degrees (convert in function argument)

    # list of layer thicknesses in nm
    d_list = [inf, 1500, 300, 10000, inf]
    # list of refractive indices
    n_list_med = [1.5, 1.5+0.1j, 1, 1.4, 3]
    n_list_bulk = [1.5, 1.5+0.1j, 1, 1, 1]
    # TODO zoran: bulk just gives 1 for all layers in E_avg - only when no imaginary component to n
    # TODO zoran: cos theta weighting present in both medium and bulk cancels => no effect => not needed?

    # n range of sensing medium
    nrange = linspace(1, 3, num=40)
    # layer which is the sensing medium
    medium = 4

    # Initialise arrays
    data_s = np.zeros(len(d_list[1:-1]))
    data_p = np.zeros(len(d_list[1:-1]))
    y_E_avg = np.zeros(len(nrange))
    x_n = np.zeros(len(nrange))

    runs = 0
    for i, n in enumerate(nrange):
        n_list_med[medium] = n
        for lambda_vac in lambda_list:
            for th in th_0:
                runs += 1
                a = TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 's')['E_avg'] +\
                    TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 's', reverse=True)['E_avg']
                b = TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 's')['E_avg'] +\
                    TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 's', reverse=True)['E_avg']
                data_s += (a/b)
                c = TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 'p')['E_avg'] +\
                     TransferMatrix(d_list, n_list_med, lambda_vac, th * degree, 'p')['E_avg']
                d = TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 'p')['E_avg'] +\
                    TransferMatrix(d_list, n_list_bulk, lambda_vac, th * degree, 'p', reverse=True)['E_avg']
                data_p += (c/d)
        data_s /= runs  # Normalise again - average over loops
        data_p /= runs  # Normalise again
        y_E_avg[i] = (data_s[0] + data_p[0]) / 2  # Take average for unpolarised light
        x_n[i] = n
    # ----------------------- END -----------------------------

    plt.figure()
    plt.plot(x_n,y_E_avg, 'o-')
    plt.xlabel('Refractive index of medium')
    plt.ylabel('Average |E|$^2$Intensity in the erbium layer per unit depth')
    plt.title('E-Field Intensity in Erbium Layer due to change in n of sensing medium')
    plt.savefig('figs/graph.png')
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

test()
