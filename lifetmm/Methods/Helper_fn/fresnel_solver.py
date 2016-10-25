from numpy import sin, cos, deg2rad, rad2deg, arcsin, arange, allclose
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

__author__ = 'mn14tm'

# Description:
# Program to solve the fresnel equations for reflectivity and transmission
# for non-magnetic, isotropic media

ni = 3.48  # Refractive index of incident light medium
nt = 1  # Refractive index of transmitted light medium
i = deg2rad(np.linspace(0, 90, 900))  # Angle of incidence (radians)

# Calculate angle of transmittance from Snell's Law
t = sp.arcsin(ni * sin(i) / nt)

# For perpendicular polarization (s)
r_s = (ni * cos(i) - nt * cos(t)) / (ni * cos(i) + nt * cos(t))
R_s = r_s ** 2

t_s = (2 * ni * cos(i)) / (ni * cos(i) + nt * cos(t))
T_s = t_s ** 2 * ((nt * cos(t)) / (ni * cos(i)))

# For parallel polarization (p)
r_p = (nt * cos(i) - ni * cos(t)) / (ni * cos(t) + nt * cos(i))
R_p = r_p ** 2
t_p = (2 * ni * cos(i)) / (ni * cos(t) + nt * cos(i))
T_p = t_p ** 2 * ((nt * cos(t)) / (ni * cos(i)))

# Calculate average for unpolarized light
R_ave = (R_p + R_s) / 2
T_ave = (T_p + T_s) / 2

plt.figure(1)
plt.subplot(121)
plt.plot(rad2deg(i), R_s, label="Perpendicular")
plt.plot(rad2deg(i), R_p, label="Parallel")
plt.plot(rad2deg(i), R_ave, '--', label="Unpolarized")
plt.ylabel('Reflectance')
plt.xlabel('Angle of incidence (degrees)')
plt.legend(loc=(0.05, 0.8), prop={'size': 13})

info = 'ni = %f \nnt = %f' % (ni, nt)
plt.text(10, 0.4, info, fontsize=12)

plt.subplot(122)
plt.plot(rad2deg(i), T_s, label="Perpendicular")
plt.plot(rad2deg(i), T_p, label="Parallel")
plt.plot(rad2deg(i), T_ave, '--', label="Unpolarized")
plt.ylabel('Transmittance')
plt.xlabel('Angle of incidence (degrees)')
plt.legend(loc=(0.05, 0.05), prop={'size': 13})

plt.show()
