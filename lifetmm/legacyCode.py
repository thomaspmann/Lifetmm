 def purcell_factor_z(self, x):
    # Insert Pseudo layers for ambient and substrate
    self._prepare_struct()

    d_list = self.d_list
    n_list = self.n_list
    m = np.where(self.m_list == True)[0][0]
    n_a = self.n_a

    def func(th_m, x):
        # Corresponding emission angle in superstrate
        th_1 = self.snell(n_list[m], n_list[0], th_m)
        # Corresponding emission angle in substrate
        th_s = self.snell(n_list[m], n_list[-1], th_m)

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
            h_term = self.H_term(th_1, th_s)
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
            h_term = self.H_term(th_1, th_s)
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
            h_term = self.H_term(th_1, th_s)
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
            h_term = self.H_term(th_1, th_s)
            last_term = n_list[m]**2 * cos(th_m) * sin(th_m)
            U_q = first_term*h_term*last_term
            return U_j + U_q

    # Evaluate upper bound of integration limit
    th_critical = self.thetaCritical(m, n_list)

    result = 0
    for pol in ['s', 'p']:
        self.set_polarization(pol)
        y, error = integrate.quad(func, 0, th_critical, args=(x,), epsrel=1E-3)
        result += (y/2)
    return result

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
    qj = self.q(n[m], n[0], self.th)
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