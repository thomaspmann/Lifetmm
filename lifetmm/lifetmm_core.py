import scipy as sp
import numpy as np

from numpy import pi, exp, sin, cos, inf, sqrt
# from scipy.interpolate import interp1d
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


def H(theta):
    if np.isreal(theta):
        return 1
    else:
        return 0


def H_term(th_1, th_s):
    # Evaluate H bracketed weighting term
    return (H(th_1) + H(th_s)) / (H(th_1)*cos(th_1) + H(th_s)*cos(th_s)).real


def prepareStruct(d_list, n_list, m, Lz=1E5):
    """ Insert pseudo layers of the ambient and substrate layers into the structure. Used for averaging.
    """
    m += 1

    # TODO: check with zoran it is OK to neglect absorption. This might be why only ok for weakly absorbing medium
    d1 = Lz/(2*n_list[0].real)
    ds = Lz/(2*n_list[-1].real)

    d_list = np.insert(d_list, 1, [d1])
    d_list = np.insert(d_list, -1, [ds])

    n_list = np.insert(n_list, 1, n_list[0])
    n_list = np.insert(n_list, -1, n_list[-1])

    return d_list, n_list, m


class LifetimeTmm:
    def __init__(self, d_list, n_list, x_step=1):
        """
        Initilise with the structure of the material to be simulated
        """
        # convert lists to numpy arrays if they're not already.
        n_list = np.array(n_list, dtype=complex)
        d_list = np.array(d_list, dtype=float)

        # input tests
        if (d_list[0] != inf) or (d_list[-1] != inf):
            raise ValueError('d_list must start and end with inf!')
        if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
            raise ValueError("Problem with n_list or d_list!")
        if type(x_step) != int:
            raise ValueError('x_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')

        # Set first and last layer thicknesses to zero (from inf) - helps for summing later in program
        d_list[0] = 0
        d_list[-1] = 0
        # # Start position of each layer
        d_cumsum = np.cumsum(d_list)
        num_layers = np.size(d_list)

        self.d_list = d_list
        self.n_list = n_list
        self.x_step = x_step
        self.d_cumsum = d_cumsum
        self.num_layers = num_layers

    def set_active_layer(self, m):
        self.m = m

    def set_wavelength(self, lam_vac):
        # input tests
        if hasattr(lam_vac, 'size') and lam_vac.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation at a time (1 wavelength, 1 angle, etc.)')
        self.lam_vac = lam_vac

    def set_bulk_refract(self, n_a):
        self.n_a = n_a

    def set_polarization(self, pol):
        self.pol = pol

    def set_angle(self, th, units='radians'):
        if hasattr(th, 'size') and th.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation at a time (1 wavelength, 1 angle, etc.)')
        if th >= pi/2 or th <= -pi/2:
            raise ValueError('The light is not incident on the structure. Check input theta '
                             '(0 <= theta < pi/2')
        if units == 'radians':
            self.th = th
        elif units == 'degrees':
            self.th = th * pi / 180
        else:
            raise ValueError('Units of angle not recognised. Please enter \'radians\' or \'degrees\'.')

    def I_mat(self, nj, nk):

        pol = self.pol
        th = self.th
        n0 = self.n_list[0]

        # transfer matrix at an interface
        qj = sqrt(nj**2 - n0.real**2 * sin(th)**2)
        qk = sqrt(nk**2 - n0.real**2 * sin(th)**2)

        if pol == 's':
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)

        elif pol == 'p':
            r = (qj * nk**2 - qk * nj**2) / (qj * nk**2 + qk * nj**2)
            t = (2 * nj * nk * qj) / (qj * nk**2 + qk * nj**2)

        else:
            raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                             " not 90$\degree$s")

        if t == 0:
            raise ValueError('Transmission is zero, cannot evaluate I+mat. Check input parameters.')

        return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, nj, dj):
        lam_vac = self.lam_vac
        n0 = self.n_list[0]
        th = self.th
        qj = sp.sqrt(nj**2 - n0.real**2 * np.sin(th)**2)
        eps = (2*pi*qj) / lam_vac

        # TODO: when eps is imaginary and dj is large the exponent becomes v. large and causes a crash.
        return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)

    def transfer_matrix(self):
        d_list = self.d_list
        n_list = self.n_list
        n = n_list
        lam_vac = self.lam_vac
        th = self.th
        x_step = self.x_step
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
            qj = sp.sqrt(n[layer]**2 - n[0].real**2 * np.sin(th)**2)
            eps = (2 * np.pi * qj) / lam_vac
            dj = d_list[layer]
            x_indices = np.where(x_mat == layer)
            # Calculate depth into layer
            x = x_pos[x_indices] - d_cumsum[layer - 1]
            # Calculate S_Prime
            S_prime = self.I_mat(n[0], n[1])
            for layerind in range(2, layer + 1):
                mL = self.L_mat(n[layerind - 1], d_list[layerind - 1])
                mI = self.I_mat(n[layerind - 1], n[layerind])
                S_prime = S_prime @ mL @ mI

            # Calculate S_dprime (double prime)
            S_dprime = np.eye(2)
            for layerind in range(layer, num_layers - 1):
                mI = self.I_mat(n[layerind], n[layerind + 1])
                mL = self.L_mat(n[layerind + 1], d_list[layerind + 1])
                S_dprime = S_dprime @ mI @ mL

            #  Electric Field Profile
            num = S_dprime[0, 0] * exp(-1j*eps*(dj-x)) + S_dprime[1, 0] * exp(1j*eps*(dj-x))
            den = S_prime[0, 0] * S_dprime[0, 0] * exp(-1j*eps*dj) + S_prime[0, 1] * S_dprime[1, 0] * exp(1j*eps*dj)
            E[x_indices] = num / den

            # Average E field inside the layer
            if not d_list[layer] == 0:
                E_avg[layer] = sum(abs(E[x_indices])**2) / (x_step*d_list[layer])

        # |E|^2
        E_square = abs(E[:]) ** 2

        # Store Results to structure
        self.x_pos = x_pos
        self.E = E
        self.E_avg = E_avg
        self.E_square = E_square

        return x_pos, E_square

    def calc_absorption(self):
        lam_vac = self.lam_vac
        d_list = self.d_list
        n_list = self.n_list
        num_layers = np.size(d_list)

        # Absorption coefficient in 1/cm
        absorption = np.zeros(num_layers)
        for layer in range(1, num_layers):
            absorption[layer] = (4 * np.pi * np.imag(n_list[layer])) / (lam_vac * 1.0e-7)
        return absorption

    def flip(self):
        """
        Flip the function front-to-back, to describe a(d-z) instead of a(z),
        where d is layer thickness.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.m = len(self.d_list)-self.m-1

    def system_matrix(self):
        d_list = self.d_list
        n_list = self.n_list
        n = n_list
        num_layers = self.num_layers

        # calculate the total system transfer matrix S
        S = self.I_mat(n[0], n[1])
        for layer in range(1, num_layers - 1):
            mL = self.L_mat(n[layer], d_list[layer])
            mI = self.I_mat(n[layer], n[layer + 1])
            # S = S @ mL @ mI
            S = S @ mL
            S = S @ mI
        S = np.real_if_close(S)
        self.S = S
        return S

    def calc_R(self):
        self.system_matrix()
        S = self.S
        R = abs(S[1, 0] / S[0, 0]) ** 2
        return R

    def calc_T(self):
        self.system_matrix()
        S = self.S
        T = abs(1 / S[0, 0]) ** 2  # note this is incorrect https://en.wikipedia.org/wiki/Fresnel_equations
        return T

    def layer_E_Field(self, m, result='E'):
        d_list = self.d_list
        n_list = self.n_list
        n = n_list
        num_layers = self.num_layers
        lam_vac = self.lam_vac
        th = self.th
        x_step = self.x_step

        # calculate primed transfer matrices for info on field inside the structure layer
        qj = sqrt(n[m]**2 - n[0].real**2 * sin(th)**2)
        eps = (2 * np.pi * qj) / lam_vac
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
            # |E|^2
            E_square = abs(E[:]) ** 2
            return x, E_square
        elif result == 'E':
            return x, E
        elif result == 'E_avg':
            # Average E field inside the layer
            E_avg = sum(abs(E)**2) / (x_step*dj)
            return E_avg
        else:
            print('Invalid result chosen. Options are \'E_square\', \'E\' and \'E_avg\'.')

    def Mj2(self, th_1, th_s, c1, d1, cs, ds):
        n1 = self.n_list[0]
        ns = self.n_list[-1]
        return 2 / (n1*H(th_1)*(abs(c1)**2 + abs(d1)**2) + ns*H(th_s)*(abs(cs)**2 + abs(ds)**2))

    def purcell_factor(self):
        d_list = self.d_list
        n_list = self.n_list
        n = n_list
        m = self.m
        n_a = self.n_a

        # Insert Pseudo layers for ambient and substrate
        # d_list, n_list, m = prepareStruct(d_list, n_list, m)

        # Evaluate upper bound of integration limit
        th_critical = thetaCritical(m, n_list)
        # Range of emission angles in active layer to evaluate over
        resolution = 5
        th_emission = np.linspace(0, th_critical, resolution, endpoint=False)
        integral = np.zeros((resolution, d_list[m]), dtype=complex)
        for pol in ['s', 'p']:
            print('\nSolving for polarisation %s' % pol)
            self.set_polarization(pol)
            for i, th_m in tqdm(enumerate(th_emission)):
                # Corresponding emission angle in superstrate
                th_1 = snell(n_list[m], n_list[0], th_m)
                # Corresponding emission angle in substrate
                th_s = snell(n_list[m], n_list[-1], th_m)

                if np.iscomplex(th_s):
                    self.set_angle(th_1)
                    # Evaluate E(x)**2 inside active layer
                    x, u_z = self.layer_E_Field(self.m)
                    S = self.system_matrix()
                    c1 = 1
                    d1 = S[1, 0] / S[0, 0]
                    cs = 1 / S[1, 1]
                    ds = 0

                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[i, :] += first_term*h_term*last_term
                elif np.iscomplex(th_1):
                    self.flip()
                    self.set_angle(th_s)
                    # Evaluate E(x)**2 inside active layer
                    x, u_z = self.layer_E_Field(self.m)
                    # Flip field back to forward direction
                    u_z = u_z[::-1]
                    S = self.system_matrix()
                    c1 = 0
                    d1 = 1 / S[1, 1]
                    cs = S[1, 0] / S[0, 0]
                    ds = 1
                    self.flip()
                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1, d1, cs, ds) * (abs(u_z)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[1, :] += first_term*h_term*last_term
                else:
                    # First mode j (D_s = 0)
                    self.set_angle(th_1)
                    # Evaluate E(x)**2 inside active layer
                    x, u_j = self.layer_E_Field(self.m)
                    S = self.system_matrix()
                    c1j = 1
                    d1j = S[1, 0] / S[0, 0]
                    csj = 1 / S[1, 1]
                    dsj = 0

                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1j, d1j, csj, dsj) * (abs(u_j)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[i, :] += first_term*h_term*last_term

                    # Second mode q (C_s = 0)
                    self.flip()
                    self.set_angle(th_s)
                    # Evaluate E(x)**2 inside active layer
                    x, v_p = self.layer_E_Field(self.m)
                    # Flip field back to forward direction
                    v_p = v_p[::-1]
                    S = self.system_matrix()
                    c1p = 0
                    d1p = 1 / S[1, 1]
                    csp = S[1, 0] / S[0, 0]
                    dsp = 1
                    self.flip()

                    # Make orthogonal to other mode
                    num = np.conjugate(c1j)*c1p + np.conjugate(d1j)*d1p
                    den = abs(c1j)**2 + d1j**2 + (n[-1]/n[0])*abs(csj)**2
                    b = num / den
                    u_q = b*u_j + v_p

                    first_term = (3/(2*n_a)) * self.Mj2(th_1, th_s, c1p, d1p, csp, dsp) * (abs(u_q)**2/3)
                    h_term = H_term(th_1, th_s)
                    last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
                    integral[i, :] += first_term*h_term*last_term

        integral = integral.real

        from scipy import integrate
        result = integrate.simps(integral, th_emission, axis=0) / 2
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
                linewidth=2,
                label=layer_text,
            )
            ax.add_patch(p)
        # ax.legend(loc='best')

        from collections import OrderedDict
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        ax.set_xlim([0, d_cumsum[-1]])
        ax.set(xlabel=r'x', ylabel=r'A.U.')
        plt.show()
