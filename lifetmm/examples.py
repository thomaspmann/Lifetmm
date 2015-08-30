from __future__ import division, print_function, absolute_import

from .lifetmm_core import *

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# To run a sample use the following in python console:

# import lifetmm.examples; lifetmm.examples.sample1()


def sample1():
    # list of layer thicknesses in nm
    d_list = [inf,100,300,200,100]
    # list of refractive indices
    n_list = [1,1.6,1,1.4+0.3j,1]
    # list of wavelength
    lambda_list = 1550 #in nm
    # incoming light angle
    th_0 = 0
    # polarization of incoming light. "s", "p" or "u"
    # TODO rewrite function to allow for unpolarised light input
    pol = "s"

    data = TransferMatrix(d_list, n_list, lambda_list, th_0, pol)

    plt.figure()
    plt.plot(data['x_pos'], data['E_square'])
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')

    # print(data['E_square'])

def sample2():
    """
    sum E field over incident theta and wavelengths
    """

# ---------------- SET UP DEVICE STRUCTURE ----------------
# list of layer thicknesses in nm
d_list = [0, 100, 300, 200, 100]
# list of refractive indices
n_list = [1, 1.6, 1, 1.4+0.3j, 1]
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
        x_pos = sample_T1(lambda_vac, th)['x_pos']
data /= (len(lambda_list)*len(th_0))  # Normalise again
# ----------------------- END -----------------------------


# ---------------------- DO PLOTS --------------------------
plt.figure()
plt.plot(x_pos, data)
plt.xlabel('Position in Device (nm)')
plt.ylabel('Normalized |E|$^2$Intensity')
plt.title('E-Field Intensity in Device')
# ----------------------- END -----------------------------

def sample3():
    """
    Example of varying seeing how the E field varies with wavelength
    """

    # list of layer thicknesses in nm
    d_list = [inf,100,300,200,100]
    # list of refractive indices
    n_list = [1,1.6,1,1.4+0.3j,1]

    # list of wavelengths to evaluate
    lambda_list = [600, 1550] #in nm

    # Set up sample structure and materials
    sample_T1 = LifetimeTmm(d_list, n_list)

def samplePol():
    # list of layer thicknesses in nm
    d_list = [inf, 100, 100]
    # list of refractive indices
    n_list = [1, 1, 1.5]
    # list of wavelengths to evaluate
    lambda_list = 1550 # in nm
    # incoming light angle (optional)
    th_0 = 0

    sample_T1 = LifetimeTmm(d_list, n_list)
    data = sample_T1(lambda_list, th_0)

    plt.figure()
    plt.plot(data['x_pos'], data['E_square'])
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('E-Field Intensity in Device')

