import numpy as np
import scipy.sparse as spr


def canonical_Rn(n, i):
    a = np.zeros(n)
    a[i] = 1
    return a


def canonical_Mn(n, i, j):
    a = np.zeros((n, n))
    a[i, j] = 1
    return spr.bsr_matrix(a)


def constant_line_one(n, i):
    return spr.bsr_matrix(np.outer(canonical_Rn(n, i), np.ones(n)))


class LinearDynamics:

    @staticmethod
    def p_pol_mul(alpha, alphap, beta, betap, omega, b, f, r):
        return np.array([1,
                         -(2 - alpha - betap - beta + alphap * beta / b),
                         (1 - alpha) * (1 - beta - betap) + alpha * beta + (beta + betap) * alphap / b,
                         -alpha * (beta + betap)])

    @staticmethod
    def p_pol(alpha, alphap, beta, betap, omega, b, f, r):
        return np.array([1,
                         -(2 - alpha - betap - beta + alphap * beta / b),
                         (1 - alpha) * (1 - beta - betap) + alpha * beta + (beta + betap) * alphap / b,
                         -alpha * (beta + betap), 0]) - \
               (1 - f) * (1 + r) * np.array([0,
                                             1,
                                             -(2 - alpha - betap - beta + alphap * beta / b),
                                             (1 - alpha) * (1 - beta - betap) + alpha * beta + (
                                                     beta + betap) * alphap / b,
                                             -alpha * (beta + betap)])

    @staticmethod
    def q_pol(alpha, alphap, beta, betap, omega, b, f, r, phi):
        alphap_f = alphap * (1 - (1 + r) * (1 - f))
        tau = phi * (1 + r) * (1 - f) / (phi + 1 - (1 + r) * (1 - f))
        tau_over_one_minus_f = phi * (1 + r) / (phi + 1 - (1 + r) * (1 - f))
        rho = -omega * beta * (1 + r - tau_over_one_minus_f) + beta * (1 - (1 + r) * (1 - f)) * (
                alpha * tau_over_one_minus_f - alphap * (1 + r))
        prefac = tau * (1 - (1 + r) * (1 - f)) * (beta + betap) + (1 - f) * rho
        return f * (np.array([0,
                              beta * (omega + alphap_f),
                              -(beta + betap) * (alphap_f + omega),
                              0, 0]) - (1 + r) * (1 - f) * np.array([0, 0,
                                                                     beta * (omega + alphap_f),
                                                                     -(beta + betap) * (alphap_f + omega),
                                                                     0]) +
                    np.array([0, 0,
                              prefac,
                              (1 - alpha) * prefac,
                              0])
                    ) / (b * (1 - (1 + r) * (1 - f)))

    def __init__(self, e):
        # Underlying economy
        self.eco = e
        self.eco.set_quantities()
        self.n = self.eco.n
        self.alpha = self.eco.firms.alpha
        self.alphap = self.eco.firms.alpha_p
        self.beta = self.eco.firms.beta
        self.betap = self.eco.firms.beta_p
        self.omega = self.eco.firms.omega
        self.z = self.eco.firms.z
        self.tau = self.eco.house.phi * (1 + self.eco.house.r) * \
                   (1 - self.eco.house.f) / (self.eco.house.phi + 1 - (1 + self.eco.house.r) * (1 - self.eco.house.f))

        self.tau_over_one_minus_f = self.eco.house.phi * (1 + self.eco.house.r) \
                                    / (self.eco.house.phi + 1 - (1 + self.eco.house.r) * (1 - self.eco.house.f))

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

        self.outer = spr.bsr_matrix(np.outer(np.ones(self.n), self.eco.j0 *
                                             np.power(self.eco.g_eq,
                                                      (1 - self.eco.b) / self.eco.b),
                                             ))
        self.p_over_g = spr.bsr_matrix(np.diag(self.eco.p_eq / self.eco.g_eq))
        self.g_over_p = spr.bsr_matrix(np.diag(self.eco.g_eq / self.eco.p_eq))

    def matrix_Q(self):
        return spr.bmat([[spr.eye(self.n)],
                         [spr.bsr_matrix(np.zeros((self.n ** 2 + 2 * self.n + 1, self.n)))]]
                        )

    def matrix_P(self):
        return spr.bmat([[spr.bsr_matrix(np.zeros((self.n, self.n))), None],
                         [None, np.eye(self.n ** 2 + 2 * self.n + 1)]])

    def forecast_block_V1(self):
        return -(self.beta + self.betap) * spr.diags(1. / self.z).dot(self.M2.transpose()) + self.betap * spr.eye(
            self.n)

    def forecast_block_W1(self):
        return (1 - self.betap) * spr.eye(self.n)

    def forecast_block_X1(self):
        return self.beta * spr.diags(self.eco.g_eq / (self.z * self.eco.p_eq)).dot(self.M2)

    def forecast_block_Y1(self):
        return - self.betap * np.sum([spr.kron(canonical_Rn(self.n, i), canonical_Mn(self.n, i, i)) / self.z[i]
                                      for i in range(self.n)], axis=0)

    def forecast_block_X2(self):
        return - (self.beta + self.betap) * spr.diags(self.eco.cons_eq / (self.z * self.eco.p_eq))

    def forecast_block_Y2(self):
        return - (self.betap + self.beta) * np.sum(
            [spr.kron(np.ones(self.n), spr.eye(self.n) - canonical_Mn(self.n, i, i)) / self.z[i]
             for i in range(self.n)], axis=0)

    def forecast_block_Z2(self):
        return (self.betap + self.beta) * self.tau_over_one_minus_f * \
               self.eco.cons_eq.reshape((self.n, 1)) / self.eco.b_eq

    def matrix_F1(self):
        return spr.bmat([[self.forecast_block_V1(), self.forecast_block_W1(), self.forecast_block_X1(),
                          self.forecast_block_Y1(), None],
                         [None, np.eye(self.n), None,
                          None, None],
                         [None, None, np.eye(self.n),
                          None, None],
                         [None, None,
                          None, np.eye(self.n ** 2), None],
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
        fst = self.alpha * self.p_over_g - self.omega * spr.diags(self.eco.p_eq).dot(self.outer) \
              / (self.eco.b * self.eco.labour_eq)
        snd = spr.diags(self.eco.p_eq / (self.z * self.eco.g_eq)).dot(
            self.alphap * self.M1.transpose() / self.eco.b - self.alpha * self.M2.transpose())
        thd = - self.alphap * spr.diags(self.eco.p_eq * self.eco.cons_eq / (self.z * self.eco.g_eq)).dot(self.outer) / \
              (self.eco.b * self.eco.b_eq)

        return fst + snd + thd

    def fixed_shortage_block_B(self):
        return - self.alpha * self.p_over_g

    def fixed_shortage_block_C(self):
        return spr.eye(self.n) - (self.alpha - self.alphap) * spr.diags(self.eco.cons_eq / (self.z * self.eco.g_eq)) \
               - self.alphap * spr.diags(1. / self.z).dot(self.M1)

    def fixed_shortage_block_D(self):
        fst = (self.alphap - self.alpha) * np.sum([self.eco.p_eq[i] * spr.kron(canonical_Mn(self.n, i, i),
                                                                               np.ones(self.n)) /
                                                   (self.z[i] * self.eco.g_eq[i])
                                                   for i in range(self.n)], axis=0)
        snd = - self.alphap * spr.diags(1. / (self.z * self.eco.g_eq)).dot(np.kron(self.eco.p_eq, spr.eye(self.n)))
        return fst + snd

    def fixed_shortage_block_E(self):
        fst = (self.alpha - self.alphap) * self.tau_over_one_minus_f * self.eco.p_eq * self.eco.cons_eq / (
                self.z * self.eco.g_eq) / self.eco.b_eq
        q = 1 + self.eco.house.r - self.tau_over_one_minus_f
        snd = -self.omega * self.eco.p_eq * q / self.eco.labour_eq
        thd = - self.alphap * q * self.eco.p_eq * self.eco.cons_eq / \
              (self.z * self.eco.g_eq * self.eco.b_eq)

        return spr.bsr_matrix((fst + snd + thd).reshape((self.n, 1)))

    def fixed_shortage_block_F(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) *
                       spr.kron(canonical_Rn(self.n, i),
                               spr.bsr_matrix(np.outer(canonical_Rn(self.n, i),
                                                       self.M2[:, i]
                                                       - self.eco.firms.z[i] * canonical_Rn(self.n, i)
                                                       - self.eco.cons_eq[i] * self.eco.j0 * np.power(self.eco.g_eq, (
                                                               1 - self.eco.b) / self.eco.b) / (
                                                               self.eco.b * self.eco.b_eq)
                                                       )
                                              )
                               )
                       for i in range(self.n)])

    def fixed_shortage_block_G(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) * self.eco.firms.z[i] *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)), canonical_Mn(self.n, i, i))
                       for i in range(self.n)])

    def fixed_shortage_block_H(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) * self.eco.cons_eq[i] *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)), canonical_Mn(self.n, i, i))
                       / self.eco.p_eq[i]
                       for i in range(self.n)])

    def fixed_shortage_block_I(self):
        return np.sum([np.exp(-self.eco.firms.sigma[i]) *
                       spr.kron(canonical_Mn(self.n, i, i), np.outer(canonical_Rn(self.n, i), np.ones(self.n)))
                       for i in range(self.n)])

    def fixed_shortage_block_J(self):
        return -(1 + self.eco.house.r) * \
               np.sum([np.exp(-self.eco.firms.sigma[i]) *
                       spr.kron(canonical_Rn(self.n, i).reshape((self.n, 1)),
                                canonical_Rn(self.n, i).reshape((self.n, 1)))
                       for i in range(self.n)]) / self.eco.b_eq

    def fixed_shortage_block_K(self):
        return (1 - self.eco.house.f) * spr.bsr_matrix(self.eco.j0 * np.power(self.eco.g_eq,
                                                               (1 - self.eco.b) / self.eco.b)) / self.eco.b

    def fixed_shortage_block_L(self):
        return (1 - self.eco.house.f) * (1 + self.eco.house.r)

    def matrix_Sf(self):
        return spr.bmat([[np.eye(self.n), None, None,
                          None, None],
                         [np.eye(self.n), None, None,
                          None, None],
                         [self.fixed_shortage_block_A(), self.fixed_shortage_block_B(), self.fixed_shortage_block_C(),
                          self.fixed_shortage_block_D(), self.fixed_shortage_block_E()],
                         [self.fixed_shortage_block_F(), self.fixed_shortage_block_G(), self.fixed_shortage_block_H(),
                          self.fixed_shortage_block_I(), self.fixed_shortage_block_J()],
                         [self.fixed_shortage_block_K(), None, None,
                          None, self.fixed_shortage_block_L()]
                         ])

    def fixed_shortage(self):
        return spr.bmat([[self.matrix_Sf(), None],
                         [self.matrix_P(), self.matrix_Q()]])

    def fixed_dynamical(self):
        return self.forecast_matrix().dot(self.fixed_shortage())
