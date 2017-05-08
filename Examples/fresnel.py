"""
Calculate the fresnel reflection from (multilayer) planar interfaces as a function of incidence angle.

[2] Principles of Nano-Optics, L. Novotny, B. Hecht
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import abs

from lifetmm.TransferMatrix import TransferMatrix


def single_interface_reflection(n1=1.47 + 0.0j, n2=1.0 + 0.0j):
    """
    Dependence of the power reflectivity and phase on the angle of incidence.
    Light incident from medium of refractive index n1 to medium of refractive index n2
    """
    # Setup simulation
    st = TransferMatrix()
    st.add_layer(0, n1)
    st.add_layer(0, n2)
    st.info()

    # Do calculations
    th_list = np.linspace(0, 90, 200, endpoint=False)
    rs_list = []
    rp_list = []
    for theta in th_list:
        st.set_incident_angle(theta, units='degrees')
        st.set_polarization('s')
        rs, t = st.calc_r_and_t()
        rs_list.append(rs)
        st.set_polarization('p')
        rp, t = st.calc_r_and_t()
        rp_list.append(rp)
    rs_list = np.array(rs_list)
    rp_list = np.array(rp_list)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, sharex='row')
    ax1.set_ylabel(r'Reflection ($|r|^2)$')
    ax1.plot(th_list, abs(rs_list) ** 2, '--', label='s')
    ax1.plot(th_list, abs(rp_list) ** 2, label='p')
    ax1.plot(th_list, (abs(rs_list) ** 2 + abs(rp_list) ** 2) / 2, label='Unpolarised')
    ax2.set_ylabel('Reflection phase (deg)')
    ax2.plot(th_list, np.angle(rs_list, deg=True), '--', label='s')
    # Note r_p is defined where the E field flips on reflection [2] pg.22
    # therefore we multiply by e^(i*pi) to shift by 180 degrees so that a negative angle implies a flipped E field
    ax2.plot(th_list, np.angle(-1 * rp_list, deg=True), label='p')
    ax2.set_xlabel('AOI (degrees)')
    ax1.legend()
    ax2.legend()
    plt.show()


def single_interface_transmission(n1=1.47 + 0.0j, n2=1.0 + 0.0j):
    """
    Dependence of the power reflectivity and phase on the angle of incidence.
    Light incident from medium of refractive index n1 to medium of refractive index n2
    """
    # Setup simulation
    st = TransferMatrix()
    st.add_layer(0, n1)
    st.add_layer(0, n2)
    st.info()

    # Do calculations
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


def gold_on_substrate():
    """ Reflection coefficient for air-gold on silica substrate (at 1.55um)"""
    # Setup simulation
    st = TransferMatrix()
    st.set_vacuum_wavelength(1550)
    # st.set_vacuum_wavelength(1550)
    st.add_layer(0, 1.6)
    st.add_layer(10, 0.52406 + 10.742j)
    st.add_layer(0, 1.6)
    st.info()

    th_list = np.linspace(0, 90, 2000, endpoint=False)
    rs_list = []
    rp_list = []
    for theta in th_list:
        # Do calculations
        st.set_incident_angle(theta, units='degrees')
        st.set_polarization('s')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rs_list.append(r)
        st.set_polarization('p')
        r, t = st.calc_reflection_and_transmission(correction=False)
        rp_list.append(r)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(th_list, rs_list, '--', label='s')
    ax.plot(th_list, rp_list, label='p')
    ax.set_xlabel('AOI (degrees)')
    ax.set_ylabel(r'Reflection ($|r|^2)$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    n1 = 1
    n2 = 0.52406 + 10.742j  # Gold @ 1550nm
    # n2 = 1.442  # Sio2 @ 1550nm

    # single_interface_reflection(n1, n2)
    # single_interface_transmission(n1, n2)
    gold_on_substrate()
