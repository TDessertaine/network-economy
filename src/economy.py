"""
The ``economy`` module
======================

This module declares the Economy class which encapsulates everything static
in the model.
This class has the network attributes (both input-output and substitution)
along with subsequent quantities(equilibrium etc). It also enhirits firms and
households attributes.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import anderson

from firms import Firms
from household import Household


warnings.simplefilter("ignore")


class Economy:
    """
    This class summarizes all the economy attributes.
    """

    @staticmethod
    def non_linear_eq_qnonzero(x, *p):
        """
        Function used for computation of equilibrium with non constant return to scale
        and general CES production function.
        :param x: guess for equilibrium
        :param p: tuple z, z_zeta, v, m_cal, zeta, q, theta, theta_bar, power
        :return: function's value at x
        """
        if len(x) % 2 != 0:
            raise ValueError("x must be of even length")

        # pylint: disable=unbalanced-tuple-unpacking
        # changements de variable: u = p_eq ; w = ((g_eq)**x) * (z**(q*zeta)) * p_eq
        u, w = np.split(x, 2)
        print(u, w)
        z_zeta, zeta, v, m_cal, q, exponent, kappa = p
        w_over_uq_p = (w /((z_zeta**q) * u))**exponent
        v1 = z_zeta * (u**zeta) * (1 - w_over_uq_p)
        m1 = m_cal * (u**zeta)
        v2 = z_zeta * w * (1 - w_over_uq_p)
        m2 = m_cal * w
        return np.array([m1 - v1 - v, m2 - v2 - kappa])

    @staticmethod
    def non_linear_eq_qzero(x, *par):
        """
        Function used for computation of equilibrium with non constant return to scale
        and Leontieff production function.
        :param x: guess for equilibrium
        :param par: tuple z_zeta, v, m_cal, q power
        :return: function's value at x
        """

        if len(x) % 2 != 0:
            raise ValueError('x must be of even length')

        # pylint: disable=unbalanced-tuple-unpacking
        p, g = np.split(x, 2)
        z, v, m_cal, exponent, kappa = par
        v1 = np.multiply(z, np.multiply(p, 1 - np.power(g, exponent)))
        m1 = np.dot(m_cal, p)
        m2 = g * m1 - p * np.dot(m_cal.T, g)
        return np.concatenate((m1 - v1 - v, m2 - g * v + kappa))

    @staticmethod
    def adj_list(j):
        """
        :param j: Network adjacency matrix.
        :return: Representation of the network as an adjacency list.
        """
        return np.array([np.where(j[i] != 0)[0] for i in range(len(j))])

    def __init__(self, n, j0, j1, a0, q, b):

        # Network initialization
        self.n = 1
        self.j1 = j1
        self.j0 = j0
        self.j_a = None
        self.a0 = a0
        a = np.array(1-a0)
        self.a = np.array([(1 - a0[i]) * a[i] / np.sum(a[i]) for i in range(n)])
        self.a_a = None
        self.q = q
        self.zeta = 1 / (q + 1)
        self.b = b
        self.coefficient = self.zeta*((self.b*self.q+1)/self.b)

        # Auxiliary network variables
        self.lamb = None
        self.a_a, self.j_a = None, None
        self.lamb_a = None
        self.m_cal = None
        self.v = None

        # Inheritances
        self.firms = None
        self.house = None

        # Equilibrium quantities
        self.p_eq = None
        self.g_eq = None
        self.mu_eq = None
        self.b_eq = None

    def init_house(self, labour, theta, gamma, phi):
        """
        Initialize a household object as instance of economy class. Cf household class.
        :param labour:
        :param theta:
        :param gamma:
        :param phi:
        :return:
        """
        self.house = Household(labour, theta, gamma, phi)

    def init_firms(self, z, sigma, alpha, alpha_p, beta, beta_p, w):
        """
        Initialize a firms object as instance of economy class. Cf firms class.
        :param z:
        :param sigma:
        :param alpha:
        :param alpha_p:
        :param beta:
        :param beta_p:
        :param w:
        :return:
        """
        self.firms = Firms(z, sigma, alpha, alpha_p, beta, beta_p, w)

    def set_house(self, house):
        """
        Sets a household object as instance of economy class.
        :param house:
        :return:
        """
        self.house = house

    def set_firms(self, firms):
        """
        Sets a firms object as instance of economy class.
        :param firms:
        :return:
        """
        self.firms = firms

    def save_eco(self, name):
        """
        Saves the economy in multi-indexed dataframe in hdf format.
        :param name: name of file.
        :return: None
        """
        first_index = np.concatenate((['Firms' for k in range(11)],
                                      ['Household' for k in range(4)]))
        second_index = np.concatenate(
            (['q', 'b', 'z', 'sigma', 'alpha', 'alpha_p', 'beta', 'beta_p',
              'w', 'p_eq', 'g_eq'],
             ['l', 'theta', 'gamma', 'phi']))
        multi_index = [first_index, second_index]
        values = np.vstack((self.q * np.ones(self.n),
                            self.b * np.ones(self.n),
                            self.firms.z,
                            self.firms.sigma,
                            self.firms.alpha * np.ones(self.n),
                            self.firms.alpha_p * np.ones(self.n),
                            self.firms.beta * np.ones(self.n),
                            self.firms.beta_p * np.ones(self.n),
                            self.firms.w * np.ones(self.n),
                            self.p_eq,
                            self.g_eq,
                            self.house.l * np.ones(self.n),
                            self.house.theta,
                            self.house.gamma * np.ones(self.n),
                            self.house.phi * np.ones(self.n),
                            ))

        df_eco = pd.DataFrame(values,
                              index=multi_index,
                              columns=np.arange(1, self.n + 1)
                              )
        df_eco.to_hdf(name + '_eco.h5', key='df', mode='w')
        np.save(name + '_network.npy', self.j_a)
        if self.q != 0:
            np.save(name + 'sub_network.npy', self.a_a)

    def set_j(self, j_a):
        """
        Sets a particular input-output network.
        :param j: a n by n matrix.
        :return: side effect
        """
        if j_a.shape != (self.n, self.n):
            raise ValueError('Input-output network must be of size (%d, %d)' % (self.n, self.n))

        self.j_a = j_a

    def set_a(self, a):
        """
        Sets a particular input-output network.
        :param a: a n by n matrix.
        :return: side effect
        """
        if a.shape != (self.n, self.n):
            raise ValueError('Substitution network must be of size (%d, %d)' % (self.n, self.n))
        self.a = a

    def set_quantities(self):
        """
        Sets redundant economy quantities as class instances.
        :return: side effect
        """
        if self.q == 0:
            self.lamb = self.j1
            self.a_a, self.j_a = np.hstack((self.a0, self.a)), np.hstack((self.j0, self.j1))
            self.lamb_a = self.j_a
            self.m_cal = np.diag(self.firms.z) - self.lamb
            self.v = np.array(self.lamb_a[0])
        elif self.q == np.inf:
            self.lamb = self.a
            self.a_a, self.j_a = np.hstack((self.a0, self.a)), np.hstack((self.j0, self.j1))
            self.lamb_a = self.a_a
            self.m_cal = np.eye(self.n) - self.lamb
            self.v = np.array(self.lamb_a[0])
        else:
            self.lamb = np.multiply(np.power(self.a, self.q * self.zeta),
                                    np.power(self.j1, self.zeta))
            self.a_a, self.j_a = np.hstack((self.a0, self.a)), np.hstack((self.j0, self.j1))
            self.lamb_a = np.multiply(np.power(self.a_a, self.q * self.zeta),
                                      np.power(self.j_a, self.zeta))
            self.m_cal = np.diag(np.power(self.firms.z, self.zeta)) - self.lamb
            self.v = np.array(self.lamb_a[0])

    def get_eps_cal(self):
        """
        Computes the smallest eigenvalue of the economy matrix
        :return: smallest eigenvalue
        """
        return np.min(np.real(np.linalg.eigvals(self.m_cal)))

    def set_eps_cal(self, eps):
        """
        Modifies firms instance to set smallest eigenvalue of economy matrix to given epsilon.
        :param eps: a real number
        :return: side effect
        """
        min_eig = self.get_eps_cal()
        z_n, sigma, alpha, alpha_p, beta, beta_p, w = self.firms.z + (eps - min_eig), \
                                                      self.firms.sigma, \
                                                      self.firms.alpha, \
                                                      self.firms.alpha_p, \
                                                      self.firms.beta, \
                                                      self.firms.beta_p, \
                                                      self.firms.w
        self.init_firms(z_n, sigma, alpha, alpha_p, beta, beta_p, w)
        self.set_quantities()

    def production_function(self, Q):
        """
        CES production function
        :param Q: if n firms, (n,n+1) matrix of available labour and goods for production
        :return: productions of the n firms
        """
        if self.q == 0:
            return np.power(np.min(np.ma.masked_invalid(np.divide(Q, self.j_a))),
                            self.b)
        elif self.q == np.inf:
            return np.power(np.prod(np.power(np.ma.masked_invalid(np.divide(Q, self.j_a)),
                                             self.a_a)),
                            self.b)
        else:
            return np.power(np.sum(np.ma.masked_invalid(self.a_a * np.power(self.j_a, 1. / self.q)
                                                        * np.power(Q, -1. / self.q))),
                            - self.b * self.q)

    def compute_p_net(self, prices):
        """
        Compute the network prices
        :param prices: current rescaled prices
        :return: current wage-rescaled network prices
        """
        if self.q == np.inf:
            return self.lamb_a
        else:
            return np.matmul(self.lamb_a, np.power(np.concatenate(([1], prices)), self.zeta))  
        
    def non_linear_eq_b1_qnonzero(self, x):
        """
        Function used to solve the equilibrium equations in the case
        b=1 and q<+inf
        """
        return ((self.firms.z*x)**(1/(self.q+1)))-(self.lamb_a[1]*x)-(self.lamb_a[0])
    
    def compute_eq(self):
        """
        :return: computes the equilibrium of the economy
        """
        # TODO: code COBB-DOUGLAS q=inf
        if self.b != 1:
            if self.q == 0:
                self.p_eq = self.j0*1/(self.firms.z*(self.house.kappa/self.j0)**(self.b-1)-self.j1)
                self.g_eq = (self.house.kappa/self.j0)**self.b
            elif self.q > 0 and self.q != np.inf:
                m_cal = self.firms.z**(self.zeta)-self.lamb_a[1]
                k = self.house.kappa
                init_guess_peq_zeta = self.lamb_a[0] / m_cal
                init_guess_w = np.divide(k, init_guess_peq_zeta)/m_cal

                par = ((self.firms.z**(self.zeta)),
                       self.zeta,
                       self.lamb_a[0],
                       m_cal,
                       self.q,
                       (self.b - 1) / (self.b * self.q + 1),
                        k)

                uw = anderson(lambda x: self.non_linear_eq_qnonzero(x, *par),
                              np.array([init_guess_peq_zeta, init_guess_w]),
                              M=50, f_tol=1e-15)

                # pylint: disable=unbalanced-tuple-unpacking
                u, w = np.split(uw, 2)
                self.p_eq = u
                self.g_eq = np.power(np.divide(w, np.power(self.firms.z, self.q * self.zeta) * u),
                                     self.b / (self.zeta * (self.b * self.q + 1)))
            elif self.q == np.inf:
                print("Yet to be implemented")
                
        else:
            if self.q == 0:
                self.p_eq = self.j0/(self.firms.z-self.j1)
                self.g_eq = self.house.kappa/self.j0
            elif self.q > 0 and self.q != np.inf:
                self.p_eq = ((self.firms.z**(self.zeta) - self.lamb_a[1])/(self.lamb_a[0]))**(self.q+1)
                self.g_eq = ((self.house.kappa)/((self.firms.z-self.lamb_a[1]*self.firms.z**(self.q*self.zeta))))((self.lamb_a[0])/(self.firms.z**(self.zeta) - self.lamb_a[1]))**(self.q+1)
            elif self.q == np.inf:
                print("Yet to be implemented")
        self.mu_eq = np.power(self.house.thetabar * self.house.v_phi, self.house.phi / (1 + self.house.phi))
        self.b_eq = self.house.thetabar / self.mu_eq
