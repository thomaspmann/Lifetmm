import numpy as np
import time
import scipy.integrate as integrate
from tqdm import *
from numpy import pi, sin, sum, exp
from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):
    def spe_layer(self, layer, radiative='Lower'):
        """ Evaluate the spontaneous emission rates for dipoles in a layer radiating into 'Lower' or 'Upper' modes.
        Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        if radiative == 'Upper':
            self.flip()
            layer = self.num_layers - layer - 1

        # Free space wave vector magnitude
        k0 = self.k0()

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # Note E_plus and E_minus are defined at cladding-layer boundary
            z = -z[::-1]

        # Angles of emission to simulate over.
        # Note: don't include pi/2 as then transmission and reflection do not make sense.
        resolution = 2 ** 8 + 1  # Must have this form for the integration later. Can change the power.
        theta_input, dth = np.linspace(0, pi / 2, num=resolution, endpoint=False, retstep=True)
        # Arrays to store the square of the E fields for
        # each angle of emission (rows) at each z coordinate (columns) before being integrated.
        E_TE_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)
        E_TM_p_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)
        E_TM_s_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)

        # Params for tqdm progress bar
        kwargs = {'total': resolution, 'unit': ' theta', 'unit_scale': True}
        # Evaluate all E field components for TE and TM modes looping over the emission angles.
        for i, theta in tqdm(enumerate(theta_input), **kwargs):
            # Set the angle to be evaluated
            self.set_angle(theta)

            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)

            # Check that the mode exists and is leaky
            assert k_11**2 < k0**2, ValueError('k_11 can not be larger than k0!')

            # !* TE modes *!
            self.set_polarization('TE')
            # Calculate E field within layer
            self.set_field('E')
            # E field coefficients in terms of incoming amplitude
            E_plus, E_minus = self.amplitude_coefficients(layer)
            E_TE = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            # Orthonormality condition: Normalise outgoing TE wave to medium refractive index.
            E_TE /= self.n_list[0].real

            # !* TM modes *!
            self.set_polarization('TM')
            # Calculate H field within layer
            self.set_field('H')
            # H field coefficients in terms of incoming amplitude
            H_plus, H_minus = self.amplitude_coefficients(layer)
            # Calculate the electric field component perpendicular (s) to the interface
            E_TM_s = k_11*(H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel (p) to the interface
            E_TM_p = q*(H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))

            # Check that results seem reasonable
            assert max(E_TE) < 100, ValueError('TMM Unstable.')
            assert max(E_TM_p) < 100, ValueError('TMM Unstable.')
            assert max(E_TM_s) < 100, ValueError('TMM Unstable.')

            # Take the squares of all E field components and add weighting
            E_TE_square_theta[i, :] += abs(E_TE) ** 2 * sin(theta)
            E_TM_s_square_theta[i, :] += abs(E_TM_s) ** 2 * sin(theta)
            E_TM_p_square_theta[i, :] += abs(E_TM_p) ** 2 * sin(theta)

        # Evaluate spontaneous emission rate for each z (columns) over all thetas (rows)
        spe_TE = integrate.romb(E_TE_square_theta, dx=dth, axis=0)
        spe_TM_p = integrate.romb(E_TM_p_square_theta, dx=dth, axis=0)
        spe_TM_s = integrate.romb(E_TM_s_square_theta, dx=dth, axis=0)

        # Outgoing E mode refractive index weighting (just after summation over j=0,M+1)
        for spe in [spe_TE, spe_TM_p, spe_TM_s]:
            spe *= self.n_list[0].real ** 3

        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        spe_TE *= 3/8
        nj = self.n_list[layer].real
        spe_TM_p *= 3/(8*(nj*k)**2)
        spe_TM_s *= 3/(4*(nj*k)**2)

        # Flip structure back to original orientation
        if radiative == 'Upper':
            self.flip()
            spe_TE = spe_TE[::-1]
            spe_TM_p = spe_TM_p[::-1]
            spe_TM_s = spe_TM_s[::-1]

        return {'z': z, 'spe_TE': spe_TE, 'spe_TM_s': spe_TM_s, 'spe_TM_p': spe_TM_p}

    def spe_structure(self):
        """ Evaluate the spontaneous emission rate vs z of the structure for each dipole orientation.
            Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumsum[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        # Initialise dictionary to store the results
        keys = ['spe_TE_lower',
                'spe_TE_upper',
                'spe_TM_p_upper',
                'spe_TM_p_lower',
                'spe_TM_s_upper',
                'spe_TM_s_lower']
        spe_rates = {key: np.zeros(len(z_pos), dtype=float) for key in keys}

        # Calculate emission rates for each layer
        print('Evaluating lower and upper radiative modes for each layer:')
        for layer in range(self.num_layers):
            # Print simulation information to command line
            if layer == 0:
                print('\tLayer -> lower cladding...')
            elif layer == self.num_layers - 1:
                print('\tLayer -> upper cladding...')
            else:
                print('\tLayer -> internal {0:d} / {1:d}...'.format(layer, self.num_layers-2))
            time.sleep(0.2)  # Fixes progress bar occurring before text

            # Find indices corresponding to the layer we are evaluating
            ind = np.where(z_mat == layer)

            # Calculate lower radiative modes
            spe = self.spe_layer(layer, radiative='Lower')
            spe_rates['spe_TE_lower'][ind] += spe['spe_TE']
            spe_rates['spe_TM_s_lower'][ind] += spe['spe_TM_s']
            spe_rates['spe_TM_p_lower'][ind] += spe['spe_TM_p']

            # Calculate upper radiative modes
            spe = self.spe_layer(layer, radiative='Upper')
            spe_rates['spe_TE_upper'][ind] += spe['spe_TE']
            spe_rates['spe_TM_s_upper'][ind] += spe['spe_TM_s']
            spe_rates['spe_TM_p_upper'][ind] += spe['spe_TM_p']

        # Total spontaneous emission rate for particular dipole orientation coupling to a particular mode
        spe_rates['spe_TE'] = spe_rates['spe_TE_lower'] + spe_rates['spe_TE_upper']
        spe_rates['spe_TM_s'] = spe_rates['spe_TM_s_lower'] + spe_rates['spe_TM_s_upper']
        spe_rates['spe_TM_p'] = spe_rates['spe_TM_p_lower'] + spe_rates['spe_TM_p_upper']

        return {'z': z_pos,
                'spe_rates': spe_rates}
