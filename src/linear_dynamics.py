import numpy as np
import scipy.sparse as spr


def canonical_Rn(n, i):
    a = np.zeros(n)
    a[i] = 1
    return a


def canonical_Mn(n, i, j):
    a = np.zeros((n, n))
    a[i, j] = 1
    return spr.bsr_matrix(np.outer(canonical_Rn(n, i)), np.outer(canonical_Rn(n, j)))


def constant_line_one(n, i):
    return spr.bsr_matrix(np.outer(canonical_Rn(n, i), np.ones(n)))


class LinearDynamics:

    @staticmethod
    def infinite_eps_pol1(alpha, alphap, beta, betap, sigma, b):
        e = np.exp(-sigma)
        delta_14 = betap * e + alpha * beta
        return np.poly1d(np.polyadd(np.polymul(np.array([1, -e]),
                                     np.array([1,
                                               -(2 - alpha - betap - beta + alphap * beta / b),
                                               (1 - alpha) * (1 - beta - betap) + alpha * beta + (
                                                       beta + betap) * alphap / b + e * betap * (1 + alphap / b),
                                               -alpha * (beta + betap) - betap * e])
                                     ),
                          e * np.array([
                              alpha,
                              delta_14 * (1 + alphap / b) - alpha * (1 - beta - betap + alphap * beta / b),
                              -(delta_14 + alpha * betap)])))

    @staticmethod
    def infinite_eps_pol2(alpha, alphap, beta, betap, omega, sigma, b, f, r, phi):
        L = (1 + r) * (1 - f)
        tau = phi * L / (phi + 1 - L)
        e = np.exp(-sigma)
        delta_11 = beta * (omega + alphap * (1 - L)) - betap * e * (1 - L)
        delta_13 = beta * (1 - alpha) - betap * e - beta - betap
        delta_14 = betap * e + alpha * beta
        delta_hat = tau * (beta * omega + (alpha * beta + beta + betap) * (1 - L))
        delta_tilde = tau * (omega + alpha * (1 - L))
        pol1 = np.array([-delta_11, (1 - alpha) * delta_11 - delta_13 * (omega + alphap * (1 - L)) + delta_hat,
                         -(1 - alpha) * delta_hat + delta_13 * delta_tilde])
        pol2 = np.array([(1 - L) * delta_14,
                         delta_14 * (omega - (1 - alpha - alphap) * (1 - L)) + alpha * ((1 - L) * delta_13 - delta_11),
                         alpha * delta_hat - delta_14 * delta_tilde])
        pol = np.polyadd(np.polymul([1, -e], pol1), +e * pol2)
        return np.poly1d(np.polymul([-1, 0], pol) / b)

    @classmethod
    def theoretical_eigenvalues_infinite_eps(cls, alphav, alphapv, betav, betapv, omega, sigmav, kappa, b, f, r, phi, lseq):
        if not(len(alphav) == len(alphapv) == len(betav) == len(betapv) == len(sigmav) == len(kappa)):
            raise ValueError('All input arrays must be of same length.')

        p = np.poly1d([1])
        for k in range(len(alphav)):
            p = p * cls.infinite_eps_pol1(alphav[k], alphapv[k], betav[k], betapv[k], sigmav[k], b)

        s = np.poly1d([0])
        for k in range(len(kappa)):
            s = s + kappa[k] * cls.infinite_eps_pol2(alphav[k], alphapv[k], betav[k], betapv[k], omega, sigmav[k],
                                                     b, f, r, phi) \
                * (p / cls.infinite_eps_pol1(alphav[k], alphapv[k], betav[k], betapv[k], sigmav[k], b))[0]
        s = s / (b * lseq)
        final_polynomial = np.poly1d([1, -(1 + r)*(1 - f)]) * p + s
        return final_polynomial.roots()

    def __init__(self, e):
        # Underlying economy
        self.eco = e
        self.eco.set_quantities()
        self.n = self.eco.n
        self.alpha = self.eco.firms.alpha * np.ones(self.n)
        self.alphap = self.eco.firms.alpha_p * np.ones(self.n)
        self.beta = self.eco.firms.beta * np.ones(self.n)
        self.betap = self.eco.firms.beta_p * np.ones(self.n)
        self.omega = self.eco.firms.omega
        self.z = self.eco.firms.z
        self.l = (1 + self.eco.house.r) * (1 - self.eco.house.f)
        self.tau = self.eco.house.phi * self.l / (self.eco.house.phi + 1 - self.l)

        self.tau_over_one_minus_f = self.eco.house.phi * (1 + self.eco.house.r) \
                                    / (self.eco.house.phi + 1 - self.l)
        self.q = 1 + self.eco.house.r - self.tau_over_one_minus_f
        # Auxiliary matrices
        tmp_diag1 = np.diag(np.power(self.z,
                                     self.eco.zeta))
        tmp_diag2 = np.power(tmp_diag1, self.eco.q)
        tmp_diag3 = np.diag(np.power(self.eco.p_eq,
                                     self.eco.q * self.eco.zeta))
        tmp_mat = np.dot(np.diag(np.power(self.eco.g_eq,
                                          (1 - self.eco.b) / self.eco.b)),
                         self.eco.lamb)
        self.M1 = spr.bsr_matrix(np.dot(np.dot(tmp_diag3, tmp_diag1 - tmp_mat), np.diag(1. / np.diag(tmp_diag3))))
        self.M2 = spr.bsr_matrix(np.dot(np.dot(tmp_diag2 * tmp_diag3, tmp_diag1 - tmp_mat / self.eco.b),
                                        np.diag(1. / np.diag(tmp_diag3 * tmp_diag2))))

        self.U = spr.bsr_matrix(np.outer(np.ones(self.n), self.eco.j0 *
                                         np.power(self.eco.g_eq,
                                                  (1 - self.eco.b) / self.eco.b),
                                         )
                                ) / self.eco.labour_eq
        self.p_over_g = spr.bsr_matrix(np.diag(self.eco.p_eq / self.eco.g_eq))
        self.g_over_p = spr.bsr_matrix(np.diag(self.eco.g_eq / self.eco.p_eq))

    def matrix_Q(self):
        return spr.bmat([[spr.eye(self.n)],
                         [spr.bsr_matrix(np.zeros((self.n ** 2 + 2 * self.n + 1, self.n)))]]
                        )

    def matrix_P(self):
        return spr.bmat([[spr.bsr_matrix(np.zeros((self.n, self.n))), None],
                         [None, spr.eye(self.n ** 2 + 2 * self.n + 1)]])

    def forecast_block_V1(self):
        return - spr.diags((self.beta + self.betap) / self.z).dot(self.M2.transpose()) + spr.diags(self.betap)

    def forecast_block_W1(self):
        return spr.diags(1 - self.betap)

    def forecast_block_X1(self):
        return spr.diags(self.beta * self.eco.g_eq / (self.z * self.eco.p_eq)).dot(self.M2)

    def forecast_block_Y1(self):
        return - spr.diags(self.betap).dot(
            np.sum([spr.kron(canonical_Rn(self.n, i), canonical_Mn(self.n, i, i)) / self.z[i]
                    for i in range(self.n)], axis=0))

    def forecast_block_X2(self):
        return - spr.diags((self.beta + self.betap) * self.eco.cons_eq / (self.z * self.eco.p_eq))

    def forecast_block_Y2(self):
        return - spr.diags(self.betap + self.beta).dot(np.sum(
            [spr.kron(canonical_Rn(self.n, i), spr.diags(1. / self.z).dot(spr.eye(self.n) - canonical_Mn(self.n, i, i)))
             for i in range(self.n)], axis=0))

    def forecast_block_Z2(self):
        return self.tau_over_one_minus_f * \
               ((self.betap + self.beta) * self.eco.cons_eq / self.z).reshape((self.n, 1)) / self.eco.b_eq

    def matrix_F1(self):
        return spr.bmat([[self.forecast_block_V1(), self.forecast_block_W1(), self.forecast_block_X1(),
                          self.forecast_block_Y1(), None],
                         [None, spr.eye(self.n), None,
                          None, None],
                         [None, None, spr.eye(self.n),
                          None, None],
                         [None, None,
                          None, spr.eye(self.n ** 2), None],
                         [None, None, None,
                          None, 1]
                         ])

    def matrix_F2(self):
        return spr.bmat([[spr.bsr_matrix(np.zeros((self.n, 2 * self.n))),
                          spr.hstack([self.forecast_block_X2(), self.forecast_block_Y2(),
                                      self.forecast_block_Z2()])],
                         [spr.bsr_matrix(np.zeros((self.n ** 2 + 2 * self.n + 1, 2 * self.n))),
                          spr.bsr_matrix(np.zeros((self.n ** 2 + 2 * self.n + 1, self.n ** 2 + self.n + 1)))]]
                        )

    def forecast_matrix(self):
        return spr.bmat([[self.matrix_F1(), self.matrix_F2()],
                         [self.matrix_Q().T, None]])

    def fixed_shortage_block_A(self):
        fst = spr.diags(self.alpha).dot(self.p_over_g) - self.omega * spr.diags(self.eco.p_eq).dot(self.U) \
              / self.eco.b
        snd = spr.diags(self.eco.p_eq / (self.z * self.eco.g_eq)).dot(
            spr.diags(self.alphap).dot(self.M1.transpose()) / self.eco.b - spr.diags(self.alpha).dot(
                self.M2.transpose()))
        thd = - self.eco.labour_eq * spr.diags(
            self.alphap * self.eco.p_eq * self.eco.cons_eq / (self.z * self.eco.g_eq)).dot(self.U) / \
              (self.eco.b * self.eco.b_eq)

        return fst + snd + thd

    def fixed_shortage_block_B(self):
        return - spr.diags(self.alpha).dot(self.p_over_g)

    def fixed_shortage_block_C(self):
        return spr.eye(self.n) - spr.diags((self.alpha - self.alphap) * self.eco.cons_eq / (self.z * self.eco.g_eq)) \
               - spr.diags(self.alphap / self.z).dot(self.M1)

    def fixed_shortage_block_D(self):
        fst = spr.diags((self.alphap - self.alpha) * self.eco.p_eq /
                        (self.z * self.eco.g_eq)).dot(spr.kron(np.ones(self.n),
                                                               spr.eye(self.n)))
        snd = - spr.diags(self.alphap / (self.z * self.eco.g_eq)).dot(
            np.sum([spr.kron(canonical_Rn(self.n, i),
                             np.outer(canonical_Rn(self.n, i), self.eco.p_eq))
                    for i in range(self.n)], axis=0)
        )
        return fst + snd

    def fixed_shortage_block_E(self):
        fst = self.alpha * self.tau_over_one_minus_f * self.eco.p_eq * self.eco.cons_eq / (
                self.z * self.eco.g_eq * self.eco.b_eq)

        snd = - self.omega * self.eco.p_eq * self.q / self.eco.labour_eq
        thd = - self.alphap * (1 + self.eco.house.r) * self.eco.p_eq * self.eco.cons_eq / \
              (self.z * self.eco.g_eq * self.eco.b_eq)

        return spr.bsr_matrix((fst + snd + thd).reshape((self.n, 1)))

    def fixed_shortage_block_F(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)),
                                spr.bsr_matrix(np.outer(canonical_Rn(self.n, i),
                                                        self.M2.toarray()[:, i].T
                                                        - self.eco.firms.z[i] * canonical_Rn(self.n, i)
                                                        - self.eco.cons_eq[i] * self.eco.j0 * np.power(self.eco.g_eq, (
                                                                1 - self.eco.b) / self.eco.b) / (
                                                                self.eco.b * self.eco.b_eq)
                                                        )
                                               )
                                )
                       for i in range(self.n)], axis=0)

    def fixed_shortage_block_G(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) * self.eco.firms.z[i] *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)), canonical_Mn(self.n, i, i))
                       for i in range(self.n)], axis=0)

    def fixed_shortage_block_H(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) * self.eco.cons_eq[i] *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)), canonical_Mn(self.n, i, i))
                       / self.eco.p_eq[i]
                       for i in range(self.n)], axis=0)

    def fixed_shortage_block_I(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) *
                       spr.kron(canonical_Mn(self.n, i, i), np.outer(canonical_Rn(self.n, i), np.ones(self.n)))
                       for i in range(self.n)], axis=0)

    def fixed_shortage_block_J(self):
        return -(1 + self.eco.house.r) * \
               np.sum([np.exp(-self.eco.firms.sigma[i]) * self.eco.cons_eq[i] *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)),
                                canonical_Rn(self.n, i).reshape((self.n, 1)))
                       for i in range(self.n)], axis=0) / self.eco.b_eq

    def fixed_shortage_block_K(self):
        return (1 - self.eco.house.f) * spr.bsr_matrix(self.eco.j0 * np.power(self.eco.g_eq,
                                                                              (1 - self.eco.b) / self.eco.b)) \
               / self.eco.b

    def fixed_shortage_block_L(self):
        return self.l

    def matrix_Sf(self):
        return spr.bmat([[spr.eye(self.n), None, None, None, None],
                         [spr.eye(self.n), None, None, None, None],
                         [self.fixed_shortage_block_A(), self.fixed_shortage_block_B(), self.fixed_shortage_block_C(),
                          self.fixed_shortage_block_D(), self.fixed_shortage_block_E()],
                         [self.fixed_shortage_block_F(), self.fixed_shortage_block_G(), self.fixed_shortage_block_H(),
                          self.fixed_shortage_block_I(), self.fixed_shortage_block_J()],
                         [self.fixed_shortage_block_K(), None, None, None, self.fixed_shortage_block_L()]
                         ])

    def fixed_shortage(self):
        return spr.bmat([[self.matrix_Sf(), None],
                         [self.matrix_P(), self.matrix_Q()]])

    def fixed_dynamical(self):
        return self.forecast_matrix().dot(self.fixed_shortage())

    def get_eigenvalues_fixed_dynamical(self):
        return np.linalg.eigvals(self.fixed_dynamical().toarray())

    def get_eigenvalues_theoretical_infinite_eps(self):
        return