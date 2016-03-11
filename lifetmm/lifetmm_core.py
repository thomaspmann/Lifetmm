import scipy as sp
import numpy as np
import scipy.integrate as integrate

from numpy import pi, exp, sin, cos, sqrt
from tqdm import *

def thetaCritical(m, n_list):
    """
    :param m: layer containing the emitting atom
    :param n_list: list of refractive indices of the layers
    :return: Return the angle at which TIRF occurs between the layer containing the atom and the cladding with
    the largest refractive index, or pi/2, whichever comes first.
    """
    # Evaluate largest refractive index of either cladding
    n_clad = max(n_list[0], n_list[-1])

    # Using Snell's law evaluate the critical angle or return pi/2 if does not exist
    angle = sp.arcsin(n_clad/n_list[m])
    if np.isreal(angle):
        return angle
    else:
        return pi/2


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


def q(nj, n0, th):
    return sqrt(nj**2 - n0.real**2 * sin(th)**2)


def H(theta):
    if np.isreal(theta):
        return 1
    else:
        return 0


def H_term(th_1, th_s):
    # Evaluate H bracketed weighting term
    return (H(th_1) + H(th_s)) / (H(th_1)*cos(th_1) + H(th_s)*cos(th_s)).real


class LifetimeTmm:
    def __init__(self):
        """
        Initilise with the structure of the material to be simulated
        """
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.m_list = np.array([], dtype=bool)
        self.d_cumsum = np.array([], dtype=float)
        self.num_layers = 0
        self.pol = 'u'

    def add_layer(self, d, n, active=False):
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        self.m_list = np.append(self.m_list, active)
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
                             'calculation at a time (1 wavelength, 1 angle, etc.)')

        if units == 'radians':
            if th >= pi/2 or th <= -pi/2:
                raise ValueError('The light is not incident on the structure. Check input theta '
                                 '(0 <= theta < pi/2')
            self.th = th
        elif units == 'degrees':
            if th >= 90 or th <= -90:
                raise ValueError('The light is not incident on the structure. Check input theta '
                                 '(0 <= theta < pi/2')
            self.th = th * pi / 180
        else:
            raise ValueError('Units of angle not recognised. Please enter \'radians\' or \'degrees\'.')
        # self.S_primed_matrix() #TODO any advantage creating separate s primed function?

    def I_mat(self, nj, nk):
        pol = self.pol
        th = self.th
        n0 = self.n_list[0]

        qj = q(nj, n0, th)
        qk = q(nk, n0, th)

        if pol in ['s', 'u']:
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        elif pol in 'p':
            r = (qj * nk**2 - qk * nj**2) / (qj * nk**2 + qk * nj**2)
            t = (2 * nj * nk * qj) / (qj * nk**2 + qk * nj**2)
        else:
            raise ValueError('Error with polarisation when calculating I_mat.')
        if t == 0:
            raise ValueError('Transmission is zero, cannot evaluate I+mat. Check input parameters.')
        return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, nj, dj):
        lam_vac = self.lam_vac
        n0 = self.n_list[0]
        qj = q(nj, n0, self.th)
        eps = (2*pi*qj) / lam_vac
        # TODO: when eps is imaginary and dj is large the exponent becomes v. large and causes a crash. Factor out 2pi?
        delta = eps*dj
        return np.array([[exp(-1j*delta), 0], [0, exp(1j*delta)]], dtype=complex)

    def system_matrix(self):
        d_list = self.d_list
        n_list = self.n_list
        # calculate the total system transfer matrix S
        S = self.I_mat(n_list[0], n_list[1])
        for layer in range(1, self.num_layers - 1):
            mL = self.L_mat(n_list[layer], d_list[layer])
            mI = self.I_mat(n_list[layer], n_list[layer + 1])
            S = S @ mL @ mI
        return S

    def calc_R_and_T(self):
        S = self.system_matrix()
        R = abs(S[1, 0] / S[0, 0]) ** 2
        T = abs(1 / S[0, 0]) ** 2  # note this is incorrect https://en.wikipedia.org/wiki/Fresnel_equations
        return R, T

    def calc_absorption(self):
        # Absorption coefficient in 1/cm
        absorption = np.zeros(self.num_layers)
        for layer in range(1, self.num_layers):
            absorption[layer] = (4*pi*np.imag(self.n_list[layer])) / (self.lam_vac*1.0e-7)
        return absorption

    def structure_E_field(self, x_step=1, result='E'):

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

        # calculate primed transfer matrices for info on field inside the structure
        E = np.zeros(len(x_pos), dtype=complex)  # Initialise E field
        E_avg = np.zeros(num_layers)

        for layer in range(1, num_layers):
            # Calculate S_Prime
            S_prime = self.I_mat(n[0], n[1])
            for layer_ind in range(2, layer + 1):
                mL = self.L_mat(n[layer_ind - 1], d_list[layer_ind - 1])
                mI = self.I_mat(n[layer_ind - 1], n[layer_ind])
                S_prime = S_prime @ mL @ mI

            # Calculate S_dprime (double prime)
            S_dprime = np.eye(2)
            for layer_ind in range(layer, num_layers - 1):
                mI = self.I_mat(n[layer_ind], n[layer_ind + 1])
                mL = self.L_mat(n[layer_ind + 1], d_list[layer_ind + 1])
                S_dprime = S_dprime @ mI @ mL

            #  Electric Field Profile
            qj = q(n[layer], n[0], self.th)
            eps = (2*pi*qj) / lam_vac
            dj = d_list[layer]
            x_indices = np.where(x_mat == layer)
            x = x_pos[x_indices] - d_cumsum[layer - 1]  # Calculate depth into layer
            num = S_dprime[0, 0] * exp(-1j*eps*(dj-x)) + S_dprime[1, 0] * exp(1j*eps*(dj-x))
            den = S_prime[0, 0] * S_dprime[0, 0] * exp(-1j*eps*dj) + S_prime[0, 1] * S_dprime[1, 0] * exp(1j*eps*dj)
            E[x_indices] = num / den

            # Average E field inside the layers
            if not d_list[layer] == 0:
                E_avg[layer] = sum(abs(E[x_indices])**2) / (x_step*d_list[layer])

        if result == 'E_square':
            E_square = abs(E[:])**2
            return x_pos, E_square
        elif result == 'E_avg':
            return E_avg
        else:
            return x_pos, E

    def layer_E_Field(self, x_step=1, result='E'):

        self._simulation_test(x_step)

        d_list = self.d_list
        n = self.n_list
        num_layers = self.num_layers
        lam_vac = self.lam_vac
        th = self.th
        m = np.where(self.m_list == True)[0][0]

        # calculate primed transfer matrices for info on field inside the structure layer
        qj = sqrt(n[m]**2 - n[0].real**2 * sin(th)**2)
        eps = (2*pi*qj) / lam_vac
        dj = d_list[m]
        x = np.arange((x_step / 2.0), dj, x_step)

        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for layerind in range(2, m + 1):
            mL = self.L_mat(n[layerind - 1], d_list[layerind - 1])
            mI = self.I_mat(n[layerind - 1], n[layerind])
            S_prime = S_prime @ mL @ mI

        # Calculate S_dprime (double prime)
        S_dprime = np.eye(2)
        for layerind in range(m, num_layers - 1):
            mI = self.I_mat(n[layerind], n[layerind + 1])
            mL = self.L_mat(n[layerind + 1], d_list[layerind + 1])
            S_dprime = S_dprime @ mI @ mL

        #  Electric Field Profile
        num = S_dprime[0, 0] * exp(-1j*eps*(dj-x)) + S_dprime[1, 0] * exp(1j*eps*(dj-x))
        den = S_prime[0, 0] * S_dprime[0, 0] * exp(-1j*eps*dj) + S_prime[0, 1] * S_dprime[1, 0] * exp(1j*eps*dj)
        E = num / den

        if result == 'E_square':
            E_square = abs(E[:])**2
            return x, E_square
        elif result == 'E_avg':
            E_avg = sum(abs(E)**2) / (x_step*dj)
            return E_avg
        else:
            return x, E

    def z_E_Field(self, x, x_step=1, result='E'):
        self._simulation_test(x_step)

        d_list = self.d_list
        n = self.n_list
        lam_vac = self.lam_vac
        m = np.where(self.m_list == True)[0][0]

        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for layer_ind in range(2, m + 1):
            mL = self.L_mat(n[layer_ind - 1], d_list[layer_ind - 1])
            mI = self.I_mat(n[layer_ind - 1], n[layer_ind])
            S_prime = S_prime @ mL @ mI

        # Calculate S_dprime (double prime)
        S_dprime = np.eye(2)
        for layer_ind in range(m, self.num_layers - 1):
            mI = self.I_mat(n[layer_ind], n[layer_ind + 1])
            mL = self.L_mat(n[layer_ind + 1], d_list[layer_ind + 1])
            S_dprime = S_dprime @ mI @ mL

        #  Electric Field Profile
        qj = q(n[m], n[0], self.th)
        eps = (2*pi*qj) / lam_vac
        dj = d_list[m]
        num = S_dprime[0, 0] * exp(-1j*eps*(dj-x)) + S_dprime[1, 0] * exp(1j*eps*(dj-x))
        den = S_prime[0, 0] * S_dprime[0, 0] * exp(-1j*eps*dj) + S_prime[0, 1] * S_dprime[1, 0] * exp(1j*eps*dj)
        E = num / den

        if result == 'E_square':
            E_square = abs(E)**2
            return E_square
        else:
            return E

    def Mj2(self, th_1, th_s, c1, d1, cs, ds):
        n1 = self.n_list[0]
        ns = self.n_list[-1]
        return 2 / (n1*H(th_1)*(abs(c1)**2 + abs(d1)**2) + ns*H(th_s)*(abs(cs)**2 + abs(ds)**2))

    def flip(self):
        """
        Flip the function front-to-back, to describe a(d-z) instead of a(z),
        where d is layer thickness.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.m_list = self.m_list[::-1]
        self.d_cumsum = np.cumsum(self.d_list)

    def _prepare_struct(self, Lz=1E4):
        """ Insert pseudo layers of the ambient and substrate layers into the structure. Used for averaging.
        """
        d_list = self.d_list
        n_list = self.n_list
        m_list = self.m_list
        # TODO: check with zoran it is OK to neglect absorption. This might be why only ok for weakly absorbing medium
        d1 = Lz/(2*n_list[0].real)
        ds = Lz/(2*n_list[-1].real)

        d_list = np.insert(d_list, 1, [d1])
        d_list = np.insert(d_list, -1, [ds])

        n_list = np.insert(n_list, 1, n_list[0])
        n_list = np.insert(n_list, -1, n_list[-1])

        m_list = np.insert(m_list, 1, False)
        m_list = np.insert(m_list, -1, False)

        self.n_list = n_list
        self.d_list = d_list
        self.m_list = m_list
        self.d_cumsum = np.cumsum(d_list)
        self.num_layers = np.size(d_list)

    def purcell_factor_layer(self):
        # Insert Pseudo layers for ambient and substrate
        self._prepare_struct()
        self.show_structure()
        d_list = self.d_list
        n_list = self.n_list
        m = np.where(self.m_list == True)[0][0]
        n_a = self.n_a

        # Evaluate upper bound of integration limit
        th_critical = thetaCritical(m, n_list)
        # Range of emission angles in active layer to evaluate over.
        resolution = 2**13 + 1
        th_emission, dth = np.linspace(0, th_critical, resolution, endpoint=False, retstep=True)
        integral = np.zeros((resolution, d_list[m]), dtype=complex)
        print('\nNumber of iterations per polarisation: {}'.format(resolution))
        for pol in ['s', 'p']:
            self.set_polarization(pol)
            for i, th_m in tqdm(enumerate(th_emission)):
                # Corresponding emission angle in superstrate
                th_1 = snell(n_list[m], n_list[0], th_m)
                # Corresponding emission angle in substrate
                th_s = snell(n_list[m], n_list[-1], th_m)

                if np.iscomplex(th_s):
                    self.set_angle(th_1)
                    # Evaluate E(x)**2 inside active layer
                    x, u_z = self.structure_E_field()
                    S = self.system_matrix()
                    c1 = 1
                    d1 = S[1, 0] / S[0, 0]
                    cs = 1 / S[0, 0]
                    ds = 0
                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[i, :] += 2*first_term*h_term*last_term
                elif np.iscomplex(th_1):
                    self.flip()
                    self.set_angle(th_s)
                    # Evaluate E(x)**2 inside active layer
                    x, u_z = self.layer_E_Field()
                    # Flip field back to forward direction
                    u_z = u_z[::-1]
                    S = self.system_matrix()
                    c1 = 0
                    d1 = 1 / S[0, 0]
                    cs = S[1, 0] / S[0, 0]
                    ds = 1
                    self.flip()
                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[1, :] += 2*first_term*h_term*last_term
                else:
                    # First mode j (D_s = 0)
                    self.set_angle(th_1)
                    # Evaluate E(x)**2 inside active layer
                    x, u_j = self.layer_E_Field()
                    S = self.system_matrix()
                    c1j = 1
                    d1j = S[1, 0] / S[0, 0]
                    csj = 1 / S[0, 0]
                    dsj = 0
                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1j, d1j, csj, dsj) * (abs(u_j)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    U_j = first_term*h_term*last_term

                    # Second mode q (C_s = 0)
                    self.set_angle(th_1)  #TODO theta 1 or s? I think 1
                    # Evaluate E(x)**2 inside active layer
                    x, v_p = self.layer_E_Field()
                    S = self.system_matrix()
                    c1p = 1
                    d1p = S[1, 1] / S[0, 1]
                    csp = 0
                    dsp = 1 / S[0, 1]
                    # Make orthogonal to other mode
                    num = np.conjugate(c1j)*c1p + np.conjugate(d1j)*d1p
                    den = abs(c1j)**2 + abs(d1j)**2 + (n_list[-1]/n_list[0])*abs(csj)**2
                    b = num / den
                    u_q = b*u_j + v_p
                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1p, d1p, csp, dsp) * (abs(u_q)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    U_q = first_term*h_term*last_term

                    integral[i, :] += (U_j+U_q)

        if np.isreal(integral.all()):
            # Discard zero imaginary part
            integral = integral.real
        else:
            raise ValueError('Cannot integrate a complex number with scipy romb algorithm.')
        purcell_factor = integrate.romb(integral, dx=dth.real, axis=0) / 2
        return purcell_factor

    def purcell_factor_z(self, x):
        # Insert Pseudo layers for ambient and substrate
        self._prepare_struct()

        d_list = self.d_list
        n_list = self.n_list
        m = np.where(self.m_list == True)[0][0]
        n_a = self.n_a

        def func(th_m, x):
            # Corresponding emission angle in superstrate
            th_1 = snell(n_list[m], n_list[0], th_m)
            # Corresponding emission angle in substrate
            th_s = snell(n_list[m], n_list[-1], th_m)

            if np.iscomplex(th_s):
                self.set_angle(th_1)
                # Evaluate E(x)**2 inside active layer
                u_z = self.z_E_Field(x)
                S = self.system_matrix()
                c1 = 1
                d1 = S[1, 0] / S[0, 0]
                cs = 1 / S[1, 1]
                ds = 0

                first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                h_term = H_term(th_1, th_s)
                last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                return 2 * first_term*h_term*last_term
            elif np.iscomplex(th_1):
                self.flip()
                self.set_angle(th_s)
                # Evaluate E(x)**2 inside active layer
                x = d_list[m] - x
                u_z = self.z_E_Field(x)
                S = self.system_matrix()
                c1 = 0
                d1 = 1 / S[1, 1]
                cs = S[1, 0] / S[0, 0]
                ds = 1
                self.flip()
                first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                h_term = H_term(th_1, th_s)
                last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                return 2 * first_term*h_term*last_term
            else:
                # First mode j (D_s = 0)
                self.set_angle(th_1)
                # Evaluate E(x)**2 inside active layer
                u_j = self.z_E_Field(x)
                S = self.system_matrix()
                c1j = 1
                d1j = S[1, 0] / S[0, 0]
                csj = 1 / S[1, 1]
                dsj = 0

                first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1j, d1j, csj, dsj) * (abs(u_j)**2/3)
                h_term = H_term(th_1, th_s)
                last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                U_j = first_term*h_term*last_term

                # Second mode q (C_s = 0)
                self.flip()
                self.set_angle(th_s)
                # Evaluate E(x)**2 inside active layer
                x = d_list[m] - x
                v_p = self.z_E_Field(x)
                S = self.system_matrix()
                c1p = 0
                d1p = 1 / S[1, 1]
                csp = S[1, 0] / S[0, 0]
                dsp = 1
                self.flip()

                # Make orthogonal to other mode
                num = np.conjugate(c1j)*c1p + np.conjugate(d1j)*d1p
                den = abs(c1j)**2 + d1j**2 + (n_list[-1]/n_list[0])*abs(csj)**2
                b = num / den
                u_q = b*u_j + v_p

                first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1p, d1p, csp, dsp) * (abs(u_q)**2/3)
                h_term = H_term(th_1, th_s)
                last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                U_q = first_term*h_term*last_term
                return U_j + U_q

        # Evaluate upper bound of integration limit
        th_critical = thetaCritical(m, n_list)

        result = 0
        for pol in ['s', 'p']:
            self.set_polarization(pol)
            y, error = integrate.quad(func, 0, th_critical, args=(x,), epsrel=1E-3)
            result += (y/2)
        return result

    def show_structure(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        d_list = self.d_list
        n_list = self.n_list
        d_cumsum = self.d_cumsum

        alphas = n_list.real / max(n_list.real)

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
                # facecolor=m[i],
                linewidth=2,
                label=layer_text,
            )
            ax.add_patch(p)

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
