from __future__ import division, print_function, absolute_import

from .lifetmm_core import *

from numpy import pi, linspace, inf, array, sum
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# To run a sample use the following in python console:

# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180

def test():
    loadSamples('T1')

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
    th_0 = 30
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
    n_list = [1, 3, 1.5, 1]
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
    lambda_list = [1550] # in nm
    # incoming light angle
    th_0 = linspace(0, 90, num=90+1) # in degrees (convert in function argument)

    # ------------- DO CALCULATIONS  -----------------
    # list of layer thicknesses in nm
    d_list = [0, 1000, 1000, 0]
    # list of refractive indices
    n_list = [1, 1.5, 3, 3]

    data_s = np.array([0.]*len(d_list))
    data_p = np.array([0.]*len(d_list))
    data = np.array([0.]*len(d_list))
    store = []
    storen = []
    for n_active in np.linspace(1, 3, num=20):
        for lambda_vac in lambda_list:
            for th in th_0:
                n_list[2] = n_active
                data_s += TransferMatrix(d_list, n_list, lambda_vac, th * degree, 's')['E_avg']
                data_p += TransferMatrix(d_list, n_list, lambda_vac, th * degree, 'p')['E_avg']
                data += (data_s + data_p) / 2
                # TODO add in normalisation
        data /= (len(lambda_list)*len(th_0))  # Normalise again - divide by number of loops
        store.append(data[1])
        storen.append(n_active)
    # ----------------------- END -----------------------------

    plt.figure()
    plt.plot(storen, store, 'o-')
    plt.xlabel('Refractive index of medium')
    plt.ylabel('Average |E|$^2$Intensity in the erbium layer per unit depth')
    plt.title('E-Field Intensity in Erbium Layer due to change in n of sensing medium')


def samplePol():

    # Loop parameters
    # list of wavelengths to evaluate
    lambda_list = [1550] # in nm
    # incoming light angle
    th_0 = linspace(0, 90, num=90+1) # in degrees (convert in function argument)

    # ------------- DO CALCULATIONS  -----------------
    # list of layer thicknesses in nm
    d_list = [0, 1000, 1000, 0]
    # list of refractive indices
    n_list_med = [1, 1.5, 3, 3]
    n_list_bulk = [1, 1.5, 1.5, 1.5]
    data_s = np.array([0.]*sum(d_list))
    data_p = np.array([0.]*sum(d_list))
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

