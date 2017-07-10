"""
Calculate the fresnel reflection from (multilayer) planar interfaces as a function of incidence angle.

[2] Principles of Nano-Optics, L. Novotny, B. Hecht
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import abs

from lifetmm.TransferMatrix import TransferMatrix


def transmission_vs_angle(st):
    """
    Dependence of the power reflectivity and phase on the angle of incidence.
    Light incident from medium of refractive index n1 to medium of refractive index n2
    """
    th_list = np.linspace(0, 90, 200, endpoint=False)
    ts_list = []
    tp_list = []
    for theta in th_list:
        st.set_incident_angle(theta, units='degrees')
        st.set_polarization('s')
        rs, ts = st.calc_r_and_t()
        ts_list.append(ts)
        st.set_polarization('p')
        rp, tp = st.calc_r_and_t()
        tp_list.append(tp)
    ts_list = np.array(ts_list)
    tp_list = np.array(tp_list)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, sharex='row')
    ax1.set_ylabel(r'Transmission ($|t|^2)$')
    ax1.plot(th_list, abs(ts_list) ** 2, '--', label='s')
    ax1.plot(th_list, abs(tp_list) ** 2, label='p')
    ax1.plot(th_list, (abs(ts_list) ** 2 + abs(tp_list) ** 2) / 2, label='Unpolarised')
    ax2.set_ylabel('Transmitted phase (deg)')
    ax2.plot(th_list, np.angle(ts_list, deg=True), '--', label='s')
    # Note r_p is defined where the E field flips on reflection [2] pg.22
    # therefore we multiply by e^(i*pi) to shift by 180 degrees so that a negative angle implies a flipped E field
    ax2.plot(th_list, np.angle(-1 * tp_list, deg=True), label='p')
    ax2.set_xlabel('AOI (degrees)')
    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    lam0 = 1550

    # Material refractive index at lam0 nm
    air = 1
    sio2 = 1.442
    au = 0.52406 + 10.742j  # gold
    ag = 0.14447 + 11.366j  # silver
    al = 1.5785 + 15.658j

    # Setup simulation
    st = TransferMatrix()
    st.add_layer(0, sio2)
    st.add_layer(100, ag)
    st.add_layer(0, air)
    st.set_vacuum_wavelength(lam0)
    st.info()
    st.plot_reflectivity_vs_angle()
    # reflection_vs_angle(st)
