import numpy as np
import time
import scipy.integrate as integrate
from tqdm import *
from numpy import pi, sin, sum, exp
from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):
    def spe_layer(self, layer, radiative='Lower', partial=False, th_num=8):
        """ Evaluate the spontaneous emission rates for dipoles in a layer radiating into 'Lower' or 'Upper' modes.
        Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        # Check lower cladding has higher refractive index than the upper cladding. For partially radiative modes.
        # Can probably remove depending on how i deal with partially radiative modes.
        assert self.n_list[0] >= self.n_list[-1], \
            ValueError('Lower cladding refractive index must be higher than the upper cladding. '
                       'Consider flipping the structure.')

        # Flip the structure and solve using lower radiative equations for upper radiative modes.
        # Results are flipped back at the end of this function to give the correct orientation again.
        if radiative == 'Upper':
            self.flip()
            layer = self.num_layers - layer - 1

        # Free space wave vector magnitude
        k0 = self.k0()

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # A_plus and A_minus are defined at first cladding-layer boundary.
            # Therefore must propagate waves backwards in the first cladding.
            z = -z[::-1]

        # Angles of emission to simulate over.
        # Note: don't include pi/2 as then transmission and reflection do not make sense.
        # res for linspace must have this form for the simpsons integration later. Can change the power.
        res = 2 ** th_num + 1
        theta_input, dth = np.linspace(0, pi / 2, num=res, endpoint=False, retstep=True)

        # Arrays to store the square of the E fields for
        # Structure of arrays to hold field components of each mode
        E2_z_th = np.zeros((len(theta_input), len(z)), dtype=[('TE', 'float64'),
                                                              ('TM_p', 'float64'),
                                                              ('TM_s', 'float64')
                                                              ])

        # Structure to hold field E(z) components of each mode for each dipole orientation for a theta
        E = np.zeros(len(z), dtype=[('TE', 'complex128'),
                                    ('TM_p', 'complex128'),
                                    ('TM_s', 'complex128')
                                    ])

        # Structure to hold field E(z) components of each mode for each dipole orientation for a theta
        spe = np.zeros(len(z), dtype=[('TE', 'float64'),
                                      ('TM_p', 'float64'),
                                      ('TM_s', 'float64')
                                      ])

        # Params for tqdm progress bar
        kwargs = {'total': res, 'unit': ' theta', 'unit_scale': True}
        # Evaluate all E field components for TE and TM modes looping over the emission angles.
        for i, theta in tqdm(enumerate(theta_input), **kwargs):
            # Set the angle to be evaluated
            self.set_angle(theta)

            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)

            # Check that the mode exists
            assert k_11**2 < k0**2, ValueError('k_11 can not be larger than k0!')

            # !* TE modes *!
            self.set_polarization('TE')
            # Calculate E field within layer
            self.set_field('E')
            # E field coefficients in terms of incoming amplitude
            E_plus, E_minus = self.amplitude_coefficients(layer)
            E['TE'] = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            # Orthonormality condition: Normalise outgoing TE wave to medium refractive index.
            E['TE'] /= self.n_list[0].real

            # !* TM modes *!
            self.set_polarization('TM')
            # Calculate H field within layer
            self.set_field('H')
            # H field coefficients in terms of incoming amplitude
            H_plus, H_minus = self.amplitude_coefficients(layer)
            # Calculate the electric field component perpendicular (s) to the interface
            E['TM_s'] = k_11*(H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel (p) to the interface
            E['TM_p'] = q*(H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))

            # Check that results seem reasonable - TMM can be unstable for large z with exponentially growing waves
            assert max(E['TE']) < 100, ValueError('TMM Unstable.')
            assert max(E['TM_s']) < 100, ValueError('TMM Unstable.')
            assert max(E['TM_p']) < 100, ValueError('TMM Unstable.')

            # Take the squares of all E field components and add sin(theta) weighting
            for key in list(E.dtype.names):
                E[key] = abs(E[key]) ** 2 * sin(theta)

            # Wave vector components in upper cladding
            k_clad, q_clad, k_11 = self.wave_vector(self.num_layers - 1)

            # Split solutions into partial and fully radiative modes
            if partial:
                if np.iscomplex(q_clad) and radiative == 'Lower':
                    for key in list(E2_z_th.dtype.names):
                        E2_z_th[key][i, :] += E[key].real
            elif not partial:
                if not np.iscomplex(q_clad) and radiative == 'Lower':
                    for key in list(E2_z_th.dtype.names):
                        E2_z_th[key][i, :] += E[key].real
                elif radiative == 'Upper':
                    for key in list(E2_z_th.dtype.names):
                        E2_z_th[key][i, :] += E[key].real

        # Evaluate spontaneous emission rate for each z (columns) over all thetas (rows)
        for key in list(spe.dtype.names):
            spe[key] = integrate.romb(E2_z_th[key], dx=dth, axis=0)

        # Outgoing mode refractive index weighting (between summation over j=0,M+1 and integral -> eps_j** 3/2)
        for key in list(spe.dtype.names):
            spe[key] *= self.n_list[0].real ** 3

        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        for key in ['TE']:
            spe[key] *= 3/8
        nj = self.n_list[layer].real
        for key in ['TM_p']:
            spe[key] *= 3/(8*(nj*k)**2)
        for key in ['TM_s']:
            spe[key] *= 3/(4*(nj*k)**2)

        # Flip structure back to original orientation
        if radiative == 'Upper':
            self.flip()
            for key in list(spe.dtype.names):
                spe[key] = spe[key][::-1]

        return {'z': z,
                'spe': spe
                }

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

        # Structure to hold field spe(z) components of each mode for each dipole orientation over structure
        spe = np.zeros(len(z_pos), dtype=[('TE_total', 'float64'),
                                          ('TM_p_total', 'float64'),
                                          ('TM_s_total', 'float64'),
                                          ('TE_lower', 'float64'),
                                          ('TM_p_lower', 'float64'),
                                          ('TM_s_lower', 'float64'),
                                          ('TE_lower_full', 'float64'),
                                          ('TM_p_lower_full', 'float64'),
                                          ('TM_s_lower_full', 'float64'),
                                          ('TE_lower_partial', 'float64'),
                                          ('TM_p_lower_partial', 'float64'),
                                          ('TM_s_lower_partial', 'float64'),
                                          ('TE_upper', 'float64'),
                                          ('TM_p_upper', 'float64'),
                                          ('TM_s_upper', 'float64')
                                          ])

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
            spe_layer = self.spe_layer(layer, radiative='Lower')['spe']
            spe['TE_lower_full'][ind] += spe_layer['TE']
            spe['TM_s_lower_full'][ind] += spe_layer['TM_s']
            spe['TM_p_lower_full'][ind] += spe_layer['TM_p']

            # Calculate
            spe_layer = self.spe_layer(layer, radiative='Lower', partial=True)['spe']
            spe['TE_lower_partial'][ind] += spe_layer['TE']
            spe['TM_s_lower_partial'][ind] += spe_layer['TM_s']
            spe['TM_p_lower_partial'][ind] += spe_layer['TM_p']

            # Calculate upper radiative modes
            spe_layer = self.spe_layer(layer, radiative='Upper')['spe']
            spe['TE_upper'][ind] += spe_layer['TE']
            spe['TM_s_upper'][ind] += spe_layer['TM_s']
            spe['TM_p_upper'][ind] += spe_layer['TM_p']

        # Total lower emission rates (partial and fully radiative modes)
        spe['TE_lower'] = spe['TE_lower_full'] + spe['TE_lower_partial']
        spe['TM_p_lower'] = spe['TM_p_lower_full'] + spe['TM_p_lower_partial']
        spe['TM_s_lower'] = spe['TM_s_lower_full'] + spe['TM_s_lower_partial']

        # Total spontaneous emission rate for particular dipole orientation coupling to a particular mode
        spe['TE_total'] = spe['TE_lower'] + spe['TE_upper']
        spe['TM_s_total'] = spe['TM_s_lower'] + spe['TM_s_upper']
        spe['TM_p_total'] = spe['TM_p_lower'] + spe['TM_p_upper']

        return {'z': z_pos,
                'spe': spe}
