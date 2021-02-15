import numpy as np


class LinearDynamics:

    @staticmethod
    def canonical_Rn(n, i):
        a = np.zeros(n)
        a[i] = 1
        return a

    @staticmethod
    def canonical_Mn(n, i, j):
        a = np.zeros((n, n))
        a[i, j] = 1
        return a

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

        # Auxiliary matrices
        tmp_diag1 = np.diag(np.power(self.z,
                                     self.eco.zeta))
        tmp_diag2 = np.power(tmp_diag1, self.eco.q)
        tmp_diag3 = np.diag(np.power(self.eco.p_eq,
                                     self.eco.q * self.eco.zeta))
        tmp_mat = np.dot(np.diag(np.power(self.eco.g_eq,
                                          (1 - self.eco.b) / self.eco.b)),
                         self.eco.lamb)
        self.M1 = np.dot(np.dot(tmp_diag3, tmp_diag1 - tmp_mat), 1. / tmp_diag3)
        self.M2 = np.dot(np.dot(tmp_diag2 * tmp_diag3, tmp_diag1 - tmp_mat / self.eco.b), 1. / (tmp_diag3 * tmp_diag2))
        self.p_over_g = np.diag(self.eco.p_eq / self.eco.g_eq)
        self.g_over_p = np.diag(self.eco.g_eq / self.eco.p_eq)

    def matrix_Q(self):
        return np.block([np.eye(self.n),
                         np.zeros((self.n ** 2 + 3 * self.n + 1, self.n))]
                        )

    def matrix_P(self):
        return np.block([[np.zeros(self.n), np.zeros((self.n, self.n ** 2 + 3 * self.n + 1))],
                         [np.zeros((self.n ** 2 + 3 * self.n + 1, self.n)), np.eye(self.n ** 2 + 3 * self.n + 1)]]
                        )

    def forecast_block_V1(self):
        return -(self.beta + self.betap) * np.dot(np.diag(1. / self.z), self.M2.T) + self.betap * np.eye(self.n)

    def forecast_block_W1(self):
        return (1 - self.betap) * np.eye(self.n)

    def forecast_block_X1(self):
        return self.beta * np.dot(np.dot(np.diag(1. / self.z), self.g_over_p), self.M2)

    def forecast_block_Y1(self):
        return -self.betap * np.sum([np.kron(self.canonical_Mn(self.n, i, i), self.canonical_Rn(self.n, i)) / self.z[i]
                                     for i in range(self.n)])

    def forecast_block_X2(self):
        return - (self.beta + self.betap) * np.diag(self.eco.cons_eq / (self.z * self.eco.p_eq))

    def forecast_block_Y2(self):
        return - (self.betap + self.beta) * np.sum(
            [np.kron(self.canonical_Mn(self.n, i, i), np.ones(self.n) - self.canonical_Rn(self.n, i)) / self.z[i]
             for i in range(self.n)])

    def forecast_block_Z2(self):
        return (self.betap + self.beta) * (1. / (self.z * self.eco.p_eq)) * self.eco.house.phi * self.eco.house.f * \
               (1 + self.eco.house.r) / (
                       self.eco.house.phi + 1 - (1 + self.eco.house.r) * (1 - self.eco.house.f))

    def matrix_F1(self):
        return np.block([[self.forecast_block_V1(), self.forecast_block_W1(), self.forecast_block_X1(),
                          self.forecast_block_Y1(), np.ones((self.n, 1))],
                         [np.zeros((self.n, self.n)), np.eye(self.n), np.zeros((self.n, self.n)),
                          np.zeros((self.n, self.n ** 2)), np.zeros((self.n, 1))],
                         [np.zeros((self.n, self.n)), np.zeros((self.n, self.n)), np.eye(self.n),
                          np.zeros((self.n, self.n ** 2)), np.zeros((self.n, 1))],
                         [np.zeros((self.n ** 2, self.n)), np.zeros((self.n ** 2, self.n)),
                          np.zeros((self.n ** 2, self.n)), np.eye(self.n ** 2), np.zeros((self.n ** 2, 1))],
                         [np.zeros((1, self.n)), np.zeros((1, self.n)), np.zeros((1, self.n)),
                          np.zeros((1, self.n ** 2)), 1]
                         ])

    def matrix_F2(self):
        return np.block([[np.zeros((self.n, 2 * self.n)), np.block([self.forecast_block_X2(), self.forecast_block_Y2(),
                                                                    self.forecast_block_Z2().T])],
                         [np.zeros((self.n ** 2 + 2 * self.n + 1, 2 * self.n)),
                          np.zeros((self.n ** 2 + 2 * self.n + 1, self.n ** 2 + self.n + 1))]]
                        )

    def forecast_matrix(self):
        return np.block([[self.matrix_F1(), self.matrix_F2()],
                         [self.matrix_Q().T, np.zeros((self.n, self.n ** 2 + 3 * self.n + 1))]])

    def fixed_shortage_block_A(self):
        outer = np.outer(self.eco.j0 *
                         np.power(self.eco.g_eq,
                                  (1 - self.eco.b) / self.eco.b),
                         np.ones(self.n))
        fst = self.alpha * self.p_over_g + self.omega * np.dot(np.diag(self.eco.p_eq), outer) \
              / (self.eco.b * self.eco.labour_eq)
        snd = np.dot(np.diag(self.eco.p_eq / (self.z * self.eco.g_eq)),
                     self.alphap * self.eco.M1.T / self.eco.b - self.eco.M2.T)
        thd = - self.alphap * np.dot(np.diag(self.eco.p_eq * self.eco.cons_eq / (self.z * self.eco.g_eq)), outer) / \
              (self.eco.house.f * self.eco.b * self.eco.b_eq)

        return fst + snd + thd

    def fixed_shortage_block_B(self):
        return - self.alpha * self.p_over_g

    def fixed_shortage_block_C(self):
        return np.eye(self.n) - (self.alpha - self.alphap) * np.diag(self.eco.cons_eq / (self.z * self.eco.g_eq)) \
               - self.alphap * np.dot(np.diag(1. / self.z), self.M1)

    def fixed_shortage_block_D(self):
        fst = (self.alphap - self.alpha) * np.sum([self.eco.p_eq[i] * np.kron(self.canonical_Mn(self.n, i, i),
                                                                              np.ones(self.n)) /
                                                   (self.z[i] * self.eco.g_eq[i])
                                                   for i in range(self.n)])
        snd = - self.alphap * np.dot(np.diag(1. / (self.z * self.eco.g_eq)), np.kron(self.eco.p_eq), np.eye(self.n))
        return fst + snd

    def fixed_shortage_block_E(self):
        pref_fst = (self.alpha - self.alphap) * self.eco.house.phi * (1 + self.eco.house.r) * self.eco.house.f / \
                   (self.eco.house.phi + 1 - (1 + self.eco.house.r) * (1 - self.eco.house.f)) / \
                   self.eco.house.theta.sum()
        fst = pref_fst * self.eco.house.theta / (self.z * self.eco.g_eq)
        q = (1 + self.eco.house.r) * (1 - (1 + self.eco.house.r) *
                                                                     (1 - self.eco.house.f)) / (
                          self.eco.house.phi + 1 - (1 + self.eco.house.r) * (1 - self.eco.house.f))
        snd = self.omega * self.eco.p_eq * q
        thd = - self.alphap * q * self.eco.p_eq * self.eco.cons_eq / \
              (self.eco.house.f * self.z * self.eco.g_eq * self.eco.b_eq)

        return (fst + snd + thd).reshape((1, self.n))

    def fixed_shortage_block_F(self):
        return np.zeros((self.n ** 2, self.n))

    def fixed_shortage_block_G(self):
        return np.zeros((self.n ** 2, self.n))

    def fixed_shortage_block_H(self):
        return np.zeros((self.n ** 2, self.n))

    def fixed_shortage_block_I(self):
        return np.zeros((self.n ** 2, self.n ** 2))

    def fixed_shortage_block_J(self):
        return np.zeros((self.n ** 2, 1))

    def fixed_shortage_block_K(self):
        return (1 - self.eco.house.f) * self.eco.j0 * np.power(self.eco.g_eq, (1 - self.eco.b) / self.eco.b) / self.eco.b

    def fixed_shortage_block_L(self):
        return (1 - self.eco.house.f) * (1 + self.eco.house.r)

    def matrix_Sf(self):
        return np.block([[np.eye(self.n), np.zeros((self.n, self.n)), np.zeros((self.n, self.n)), 
                          np.zeros((self.n, self.n ** 2)), np.zeros((self.n, 1))],
                         [np.eye(self.n), np.zeros((self.n, self.n)), np.zeros((self.n, self.n)), 
                          np.zeros((self.n, self.n ** 2)), np.zeros((self.n, 1))],
                         [self.fixed_shortage_block_A(), self.fixed_shortage_block_B(), self.fixed_shortage_block_C(),
                          self.fixed_shortage_block_D(), self.fixed_shortage_block_E()],
                         [self.fixed_shortage_block_F(), self.fixed_shortage_block_G(), self.fixed_shortage_block_H(),
                          self.fixed_shortage_block_I(), self.fixed_shortage_block_J()],
                         [self.fixed_shortage_block_K(), np.ones(self.n), np.ones(self.n),
                          np.ones(self.n ** 2), self.fixed_shortage_block_L()]
                         ])

    def fixed_shortage(self):
        return np.block([[self.matrix_Sf(), np.zeros((self.n ** 2 + 4 * self.n + 1, self.n))],
                         [self.matrix_P(), self.matrix_Q()]])

    def fixed_dynamical(self):
        return np.dot(self.forecast_matrix(), self.fixed_shortage())