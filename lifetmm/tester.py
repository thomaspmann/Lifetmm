import scipy as sp
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from numpy import pi, exp, sin, cos, sqrt
from tqdm import *


class LifetimeTmm:
    def __init__(self):
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.m_list = np.array([], dtype=bool)
        self.m = 0
        self.d_cumsum = np.array([], dtype=float)
        self.num_layers = 0
        self.pol = 'u'
        self.lam_vac = 0
        self.n_a = 0
        self.th = 0

    def add_layer(self, d, n, active=False):
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        self.m_list = np.append(self.m_list, active)
        # Set m to location where active layer is
        if self.m_list.any():
            self.m = np.where(self.m_list == True)[0][0]
        self.d_cumsum = np.cumsum(self.d_list)
        self.num_layers = np.size(self.d_list)

    def set_wavelength(self, lam_vac):
        if hasattr(lam_vac, 'size') and lam_vac.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation at a time (1 wavelength, 1 angle, etc.)')
        self.lam_vac = lam_vac

    def set_bulk_n(self, n_a):
        self.n_a = n_a

    def set_polarization(self, pol):
        if pol not in ['s', 'p']:
            raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                             " not 90$\degree$s")
        self.pol = pol

    def set_angle(self, th, units='radians'):
        if hasattr(th, 'size') and th.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation for each angle at a time')

        if units == 'radians':
            if th >= pi/2 or th <= -pi/2:
                raise ValueError('The light is not incident on the structure. Check input theta '
                                 'satisfies -pi/2 <= theta < pi/2')
            self.th = th
        elif units == 'degrees':
            if th >= 90 or th <= -90:
                raise ValueError('The light is not incident on the structure. Check input theta '
                                 'satisfies -90 <= theta < 90')
            self.th = th * (pi/180)
        else:
            raise ValueError('Units of angle not recognised. Please enter \'radians\' or \'degrees\'.')

    @staticmethod
    def q(nj, n0, th):
        assert 0 <= th < pi/2, 'Error evaluating wave vector. Check input theta is between 0 and pi/2.'
        return sqrt(nj**2 - n0.real**2 * sin(th)**2)

    def I_mat(self, nj, nk):
        th = self.th
        n0 = self.n_list[0]

        qj = self.q(nj, n0, th)
        qk = self.q(nk, n0, th)

        if self.pol is 'p':
            r = (qj * nk**2 - qk * nj**2) / (qj * nk**2 + qk * nj**2)
            t = (2 * nj * nk * qj) / (qj * nk**2 + qk * nj**2)
        else:  # pol is 's' (or 'u')
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        assert t != 0, ValueError('Transmission is zero, cannot evaluate I_mat.')
        return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, nj, dj):
        qj = self.q(nj, self.n_list[0], self.th)
        eps = (2*pi*qj) / self.lam_vac
        delta = eps*dj
        return np.array([[exp(-1j*delta), 0], [0, exp(1j*delta)]], dtype=complex)

    def calc_s_matrix(self):
        d_list = self.d_list
        n = self.n_list
        num_layers = self.num_layers
        # calculate the total system transfer matrix S
        S = self.I_mat(n[0], n[1])
        for layer in range(1, num_layers - 1):
            mL = self.L_mat(n[layer], d_list[layer])
            mI = self.I_mat(n[layer], n[layer + 1])
            S = S @ mL @ mI
        return S

    def calc_s_primed(self):
        d_list = self.d_list
        n = self.n_list
        # Active layer
        m = self.m

        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for v in range(1, m):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_prime = S_prime @ mL @ mI
        # Calculate S_dPrime (doubled prime)
        S_dprime = self.I_mat(n[m], n[m + 1])
        for v in range(m + 1, self.num_layers - 1):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_dprime = S_dprime @ mL @ mI
        return S_prime, S_dprime

    def calc_R_and_T(self):
        S = self.system_matrix()
        R = abs(S[1, 0] / S[0, 0]) ** 2
        T = abs(1 / S[0, 0]) ** 2
        # note this is incorrect T: https://en.wikipedia.org/wiki/Fresnel_equations
        return R, T

    def calc_absorption(self):
        # Absorption coefficient in 1/cm
        absorption = np.zeros(self.num_layers)
        for layer in range(1, self.num_layers):
            absorption[layer] = (4*pi*np.imag(self.n_list[layer])) / (self.lam_vac*1.0e-7)
        return absorption

    def structure_E_field(self, x_step=1, time_reversal=False):

        self._simulation_test(x_step)

        d_list = self.d_list
        n = self.n_list
        lam_vac = self.lam_vac
        d_cumsum = self.d_cumsum
        num_layers = self.num_layers

        # calculate the total system transfer matrix S
        S = self.calc_s_matrix()

        # x positions to evaluate E field at over entire structure
        x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
        comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))
        x_mat = sum(comp1 > comp2, 0)

        E = np.zeros(len(x_pos), dtype=complex)
        E_avg = np.zeros(num_layers)

        for layer in range(1, num_layers - 1):
            self.m = layer
            S_prime, S_dprime = self.calc_s_primed()

            #  Electric Field Profile
            qj = self.q(n[layer], n[0], self.th)
            eps = (2*pi*qj) / lam_vac
            x_indices = np.where(x_mat == layer)
            x = x_pos[x_indices] - d_cumsum[layer - 1]  # Calculate depth into layer

            # Alternate Calculation
            if not time_reversal:
                rR = S[1, 0] / S[0, 0]
            else:
                rR = S[1, 1] / S[0, 1]
            t_plus = (S_prime[1, 1] - rR * S_prime[0, 1]) / np.linalg.det(S_prime)
            t_mimus = (rR * S_prime[0, 0] - S_prime[1, 0]) / np.linalg.det(S_prime)
            E[x_indices] = t_plus*exp(1j*eps*x) + t_mimus*exp(-1j*eps*x)

            # Average E field inside the layers
            if not d_list[layer] == 0:
                E_avg[layer] = sum(abs(E[x_indices])**2) / (x_step*d_list[layer])

        E_square = abs(E[:])**2

        return {'E': E, 'E_square': E_square, 'E_avg': E_avg}

    def layer_E_Field(self, x_step=1, time_reversal=False):

        self._simulation_test(x_step)

        d_list = self.d_list
        n = self.n_list

        # calculate the transfer matrices
        S = self.calc_s_matrix()
        S_prime, S_dprime = self.calc_s_primed()

        #  Electric Field Profile
        qj = self.q(n[self.m], n[0], self.th)
        eps = (2*pi*qj) / self.lam_vac
        dj = d_list[self.m]
        x = np.arange((x_step / 2.0), dj, x_step)

        # Alternate Calculation
        if not time_reversal:
            rR = S[1, 0] / S[0, 0]
        else:
            rR = S[1, 1] / S[0, 1]

        t_plus = (S_prime[1, 1] - rR*S_prime[0, 1]) / np.linalg.det(S_prime)
        t_minus = (rR*S_prime[0, 0] - S_prime[1, 0]) / np.linalg.det(S_prime)
        E = t_plus*exp(1j*eps*x) + t_minus*exp(-1j*eps*x)

        E_square = abs(E[:])**2
        E_avg = sum(E_square) / (x_step*dj)

        # Store matrices
        self.S = S
        self.S_prime = S_prime
        self.S_dprime = S_dprime

        return {'x': x, 'E': E, 'E_square': E_square, 'E_avg': E_avg}

    def show_structure(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        d_list = self.d_list
        n_list = self.n_list
        d_cumsum = self.d_cumsum

        # Shades to fill rectangles with based on refractive index
        alphas = abs(n_list) / max(abs(n_list))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, dx in enumerate(d_list[1:-1]):
            x = d_cumsum[i]
            layer_text = ('{0.real:.2f} + {0.imag:.2f}j'.format(n_list[i+1]))
            p = patches.Rectangle(
                (x, 0.0),    # (x,y)
                dx,          # width
                1.0,         # height
                alpha=alphas[i+1],
                linewidth=2,
                label=layer_text,
            )
            ax.add_patch(p)

        # Create legend without duplicate keys
        from collections import OrderedDict
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        ax.set_xlim([0, d_cumsum[-1]])
        ax.set(xlabel=r'x', ylabel=r'A.U.')
        plt.show()

    def _simulation_test(self, x_step):
        if (self.d_list[0] != 0) or (self.d_list[-1] != 0):
            raise ValueError('Structure must start and end with 0!')
        if type(x_step) != int:
            raise ValueError('x_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')

    @staticmethod
    def snell(n_1, n_2, th_1):
        """
        return angle theta in layer 2 with refractive index n_2, assuming
        it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
        that "angles" may be complex!!
        """
        # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
        # give different results e.g. for arcsin(2).)
        # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
        # arcsin(2) due to branch cut
        return sp.arcsin(np.real_if_close(n_1*sin(th_1) / n_2))

    def flip(self):
        """ Flip the structure front-to-back.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.m_list = self.m_list[::-1]
        self.m = np.where(self.m_list == True)[0][0]
        self.d_cumsum = np.cumsum(self.d_list)

    def thetaCritical(self):
        """
        :return: Return the angle at which TIRF occurs between the layer containing the
        atom and the cladding with the largest refractive index, or pi/2, whichever comes
        first.
        """
        n_list = self.n_list
        m = self.m
        assert len(n_list) > 2, 'Structure must have at least 3 layers; two cladding and one active.'
        assert 0 < m < len(n_list), 'Active layer can not be the cladding.'

        # Evaluate largest refractive index of either cladding
        n_clad = max(n_list[0], n_list[-1])

        # Using Snell's law evaluate the critical angle or return pi/2 if does not exist
        angle = sp.arcsin(n_clad / n_list[m])
        if np.isreal(angle):
            return angle
        else:
            return pi / 2

    def structure_ldos(self, x_step=1):
        self._simulation_test(x_step)

        d_list = self.d_list
        n = self.n_list
        lam_vac = self.lam_vac
        d_cumsum = self.d_cumsum
        num_layers = self.num_layers

        # x positions to evaluate E field at over entire structure
        x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
        comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))
        x_mat = sum(comp1 > comp2, 0)

        E = np.zeros(len(x_pos), dtype=complex)
        E_square = np.zeros(len(x_pos), dtype=complex)
        result = np.zeros(len(x_pos), dtype=complex)

        for layer in range(1, num_layers - 1):
            # Calculate x depth into layer to evaluate E field at
            x_indices = np.where(x_mat == layer)
            x = x_pos[x_indices] - d_cumsum[layer - 1]

            resolution = 2 ** 8 + 1
            th_emission, dth = np.linspace(0, pi / 2, resolution, endpoint=False, retstep=True)
            integral = np.zeros((resolution, self.d_cumsum[-1]), dtype=complex)
            # Params for tqdm progress bar
            kwargs = {
                'total': resolution,
                'unit': 'theta',
                'unit_scale': True,
                'leave': True,
            }
            for i, th in tqdm(enumerate(th_emission), **kwargs):
                self.set_angle(th)

                # calculate the total system transfer matrix S
                S = self.I_mat(n[0], n[1])
                for layer in range(1, num_layers - 1):
                    mL = self.L_mat(n[layer], d_list[layer])
                    mI = self.I_mat(n[layer], n[layer + 1])
                    S = S @ mL @ mI

                # Time reversed structure reflection for right side incoming wave
                rR = S[1, 1] / S[0, 1]

                # Calculate S_Prime
                S_prime = self.I_mat(n[0], n[1])
                for v in range(1, layer):
                    mL = self.L_mat(n[v], d_list[v])
                    mI = self.I_mat(n[v], n[v + 1])
                    S_prime = S_prime @ mL @ mI

                # Calculate S_dPrime (doubled prime)
                S_dprime = self.I_mat(n[layer], n[layer + 1])
                for v in range(layer + 1, self.num_layers - 1):
                    mL = self.L_mat(n[v], d_list[v])
                    mI = self.I_mat(n[v], n[v + 1])
                    S_dprime = S_dprime @ mL @ mI

                # Calculate E field
                qj = self.q(n[layer], n[0], self.th)
                eps = (2 * pi * qj) / lam_vac

                t_plus = (S_prime[1, 1] - rR * S_prime[0, 1]) / np.linalg.det(S_prime)
                t_mimus = (rR * S_prime[0, 0] - S_prime[1, 0]) / np.linalg.det(S_prime)

                E[x_indices] = t_plus*exp(1j*eps*x) + t_mimus*exp(-1j*eps*x)
                E_square[x_indices] = abs(E[x_indices]) ** 2

                integral[i, :] += E_square[x_indices] * sin(th)

            if np.isreal(integral.all()):
                # Discard zero imaginary part
                integral = integral.real
            else:
                raise ValueError('Cannot integrate a complex number with scipy romb algorithm.')

            result = integrate.romb(integral, dx=dth.real, axis=0)
        return result


def mcgehee():
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j)
    st.add_layer(220, 2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    # st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    # st.show_structure()

    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0)

    y = st.structure_E_field(time_reversal=True)['E_square']

    plt.figure()
    plt.plot(y)
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def mcgehee2():
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704 + 0.01161j)
    st.add_layer(35, 1.4621 + 0.04426j)
    st.add_layer(220, 2.12 + 0.3166016j, active=True)
    st.add_layer(7, 2.095 + 2.3357j)
    st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    # st.show_structure()

    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0)

    y = st.layer_E_Field()['E_square']

    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def spe():
    st = LifetimeTmm()

    st.add_layer(0, 3.48)
    st.add_layer(1500, 3.48)
    st.add_layer(1500, 1)
    st.add_layer(0, 1)

    st.set_wavelength(1540)
    st.set_polarization('s')

    y = st.structure_E_field(time_reversal=True)['E_square']

    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.show()


if __name__ == "__main__":
    # mcgehee()
    # mcgehee2()
    spe()