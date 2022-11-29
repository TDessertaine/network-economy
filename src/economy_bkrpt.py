# network-economy is a simulation program for the Network Economy ABM desbribed in <https://doi.org/10.1016/j.jedc.2022.104362>
# Copyright (C) 2020 Théo Dessertaine
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
The ``economy`` module
======================

This module declares the Economy class which encapsulates everything static in the model.
This class has the network attributes (both input-output and substitution) along with subsequent quantities
(equilibrium etc). It also inherits firms and households attributes.
"""
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import leastsq
from time import time 
import random

from firms import Firms
from household import Household
from network import create_net

warnings.simplefilter("ignore")


net_type = ['regular', 'm-regular', 'er']


class Economy:
    """

    """

    def __init__(self, n: int = None, d: int = None, netstring: str = None, directed: bool = None, j0: np.array = None, a0: np.array = None, q: float = None, b: float = None,nested_CES = [False,None],**kwargs) -> None:
        """_summary_

        Args:
            n (int > 0): Number of firms in the network. 
            d (int > 0): Average number of in and out links in the network.
            netstring (str): _description_
            directed (bool): _description_
            j0 (np.array): _description_
            a0 (np.array): _description_
            q (float): _description_
            b (float): _description_

        Returns:
            object: _description_
        """

        if not n > 0:
            raise ValueError("n must be a positive integer.")

        if not d >= 0:
            raise ValueError("d must be a positive integer.")

        if not netstring in net_type:
            raise ValueError(
                "netstring not supported. Please choose 'regular' for regular, 'm-regular' for multi-regular, 'er' for Erdös-Renyi.")

        # if j0.any() < 0:
        #     raise ValueError("Entries of j0 must be greater or equal to 0.")

        # if a0.any() < 0 or a0.any() > 1:
        #     raise ValueError("Entries of a0 must be between 0 and 1.")

        if not q >= 0:
            raise ValueError("q must be a positive real number.")

        if not b >= 0:
            raise ValueError("b must be a positive real number.")

        # Network initialization
        self.n = n if n else 100
        d = d if d else 15
        netstring = netstring if netstring else 'm_regular'
        directed = directed if directed else True

        self.j = create_net(netstring, directed, n, d)  # Network creation
        self.init_n_links = self.j.sum(axis = 1) # Number of links per firms
        self.j_save = self.j.copy()
        self.list_firms = np.arange(n)
        # self.j0 = j0 if j0 else np.ones(self.n)
        # self.a0 = a0 if a0 else 0.5 * np.ones(self.n)
        self.j0 = j0
        self.j0_save = j0.copy()
        self.a0 = a0
        self.a0_save = a0.copy()

        a = np.multiply(np.random.uniform(0, 1, (n, n)), self.j)

        self.a = np.array([(1 - a0[i]) * a[i] / np.sum(a[i])
                          for i in range(n)])
        self.a_save = self.a.copy()

        self.q = q if q else 0
        self.zeta = 1 / (q + 1)
        self.b = b if b else 1
        if nested_CES[0] : # If we are in a nested CES economy 
            self.leontief_deg =  nested_CES[1] # Degré moyen CES de l'économie.
            self.leontief_dict = [random.sample(list(np.where(self.j[i,:] == 1)[0]),self.leontief_deg) for i in range(self.n)]
            self.leontief_net = np.zeros_like(self.j)
            for i in range(self.n): self.leontief_net[i,self.leontief_dict[i]] = 1 # We create a new network matrix with only the leontief links
        elif q == 0 : 
            self.leontief_net = self.j.copy()
        # Auxiliary network variables
        self.lamb = None
        self.a_a = None
        self.j_a = None
        self.lamb_a = None
        self.m_cal = None
        self.v = None
        self.kappa = None
        self.zeros_j_a = None

        # Firms and household sub-classes
        self.firms = None
        self.house = None

        # Equilibrium quantities
        self.p_eq = None
        self.g_eq = None
        self.mu_eq = None
        self.labour_eq = None
        self.cons_eq = None
        self.b_eq = None
        self.utility_eq = None
        self.time_table_1 = []
        self.time_table_2 = []
    def clear_all(self):
        self.j = self.j_save.copy()
        self.a0 = self.a0_save.copy()
        self.j0 = self.j0_save.copy()
        self.a = self.a_save.copy()

        self.time_table_1 = []
        self.time_table_2 = []
        # Auxiliary network variables
        self.lamb = None
        self.a_a = None
        self.j_a = None
        self.lamb_a = None
        self.m_cal = None
        self.v = None
        self.kappa = None
        self.zeros_j_a = None

        # Firms and household sub-classes
        self.firms = None
        self.house = None

        # Equilibrium quantities
        self.p_eq = None
        self.g_eq = None
        self.mu_eq = None
        self.labour_eq = None
        self.cons_eq = None
        self.b_eq = None
        self.utility_eq = None
        

    def init_house(self, l_0, theta, gamma, phi, omega_p=None, f=None, r=None):
        """
        Initialize a household object as instance of economy class. Refer to household class.
        :param l_0: baseline work offer,
        :param theta: preferency factors,
        :param gamma: aversion to work parameter,
        :param phi: Frisch index,
        :param omega_p: confidence parameter
        :param f: fraction of budget to save,
        :param r: savings growth rate.
        :return: Initializes household class with given parameters.
        """
        self.house = Household(l_0, theta.copy(), gamma, phi, omega_p, f, r) # The .copy() for theta is to ensure that when we modify it and then clear all it does not change the initial input.

    def init_firms(self, z, sigma, alpha, alpha_p, beta, beta_p, omega):
        """
        Initialize a firms object as instance of economy class. Refer to firms class.
        :param z: Productivity factors,
        :param sigma: Depreciation of stocks parameters,
        :param alpha: Log-elasticity of prices' growth rates against surplus,
        :param alpha_p: Log-elasticity of prices' growth rates against profits,
        :param beta: Log-elasticity of productions' growth rates against profits,
        :param beta_p: Log-elasticity of productions' growth rates against surplus,
        :param omega: Log-elasticity of wages' growth rates against labor-market tensions.
        :return: Initializes firms class with given parameters.
        """
        self.firms = Firms(z, sigma, alpha, alpha_p, beta, beta_p, omega)

    # Setters for class instances

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

    # Update methods for firms and household

    def update_firms_z(self, z):
        self.firms.update_z(z)
        self.set_quantities()
        self.compute_eq()

    def update_firms_sigma(self, sigma):
        self.firms.update_sigma(sigma)

    def update_firms_alpha(self, alpha):
        self.firms.update_alpha(alpha)

    def update_firms_alpha_p(self, alpha_p):
        self.firms.update_alpha_p(alpha_p)

    def update_firms_beta(self, beta):
        self.firms.update_beta(beta)

    def update_firms_beta_p(self, beta_p):
        self.firms.update_beta_p(beta_p)

    def update_firms_w(self, omega):
        self.firms.update_w(omega)

    def update_house_labour(self, labour):
        self.house.update_labour(labour)
        self.compute_eq()

    def update_house_theta(self, theta):
        self.house.update_theta(theta)
        self.compute_eq()

    def update_house_gamma(self, gamma):
        self.house.update_gamma(gamma)
        self.compute_eq()

    def update_house_phi(self, phi):
        self.house.update_phi(phi)
        self.compute_eq()

    def update_house_w_p(self, omega_p):
        self.house.update_w_p(omega_p)

    def update_house_f(self, f):
        self.house.update_f(f)

    def update_house_r(self, r):
        self.house.update_r(r)

    # Setters for the networks and subsequent instances

    def set_j(self, j):
        """
        Sets a particular input-output network.
        :param j: a n by n matrix.
        :return: side effect
        """
        if j.shape != (self.n, self.n):
            raise ValueError(
                'Input-output network must be of size (%d, %d)' % (self.n, self.n))

        self.j = j
        self.init_n_links = self.j.sum(axis = 1) # Number of links per firms
        self.set_quantities()
        self.compute_eq()

    def set_a(self, a):
        """
        Sets a particular input-output network.
        :param a: a n by n matrix.
        :return: side effect
        """
        if a.shape != (self.n, self.n):
            raise ValueError(
                'Substitution network must be of size (%d, %d)' % (self.n, self.n))
        self.a = a
        self.set_quantities()
        self.compute_eq()

    def set_quantities(self):
        """
        Sets redundant economy quantities as class instances.
        :return: side effect
        """
        if self.q == 0:
            self.lamb = self.j
            self.a_a = np.hstack((np.array([self.a0]).T, self.a))
            self.j_a = np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = self.j_a
            self.m_cal = np.diag(self.firms.z) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        elif self.q == np.inf:
            self.lamb = self.a
            self.a_a = np.hstack((np.array([self.a0]).T, self.a))
            self.j_a = np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = self.a_a
            self.m_cal = np.eye(self.n) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        else:
            self.lamb = np.multiply(np.power(self.a, self.q * self.zeta),
                                    np.power(self.j, self.zeta))
            self.a_a = np.hstack((np.array([self.a0]).T, self.a))
            self.j_a = np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = np.multiply(np.power(self.a_a, self.q * self.zeta),
                                      np.power(self.j_a, self.zeta))
            self.m_cal = np.diag(np.power(self.firms.z, self.zeta)) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        self.mu_eq = np.power(np.power(self.house.gamma, 1./self.house.phi) * np.sum(self.house.theta) *
                              (1 - (1 - self.house.f) * (1 + self.house.r)) /
                              (self.house.f * np.power(self.house.l_0,
                               1 + 1./self.house.phi)),
                              self.house.phi / (1 + self.house.phi))
        self.kappa = self.house.theta / self.mu_eq
        self.zeros_j_a = self.j_a != 0

    def get_eps_cal(self):
        """
        Computes the smallest eigenvalue of the economy matrix
        :return: smallest eigenvalue
        """
        return np.min(np.real(np.linalg.eigvals(self.m_cal)))

    def set_eps_cal(self, eps):
        """
        Modifies firms instance to set smallest eigenvalue of economy matrix to given epsilon.
        :param eps: a real number,
        :return: side effect.
        """
        min_eig = self.get_eps_cal()
        z_n = self.firms.z * \
            np.power(1 + (eps - min_eig) /
                     np.power(self.firms.z, self.zeta), self.q + 1)
        sigma = self.firms.sigma
        alpha = self.firms.alpha
        alpha_p = self.firms.alpha_p
        beta = self.firms.beta
        beta_p = self.firms.beta_p
        omega = self.firms.omega
        self.init_firms(z_n, sigma, alpha, alpha_p, beta, beta_p, omega)
        self.set_quantities()
        self.compute_eq()

    def update_b(self, b):
        """
        Sets return to scale parameter
        :param b: return to scale
        :return: side effect
        """
        self.b = b
        self.compute_eq()

    def update_q(self, q):
        self.q = q
        self.zeta = 1 / (q + 1)
        self.set_quantities()
        self.compute_eq()

    def update_network(self, netstring, directed, d, n):
        self.j = create_net(netstring, directed, n, d)
        a = np.multiply(np.random.uniform(0, 1, (n, n)), self.j)
        self.a = np.array([(1 - self.a0[i]) * a[i] / np.sum(a[i])
                          for i in range(n)])
        self.set_quantities()
        self.compute_eq()

    def update_a0(self, a0):
        self.a0 = a0
        a = np.multiply(np.random.uniform(0, 1, (self.n, self.n)), self.j)
        self.a = np.array([(1 - self.a0[i]) * a[i] / np.sum(a[i])
                          for i in range(self.n)])
        self.set_quantities()
        self.compute_eq()

    def update_j0(self, j0):
        self.j0 = j0
        self.set_quantities()
        self.compute_eq()

    def production_function(self, q_available,srv_idx= None, srv_idx_a = None):
        """
        CES production function.
        :param q_available: matrix of available labour and goods for production,
        :return: production levels of the firms.
        """
        if srv_idx is None : 
            srv_idx,srv_idx_a = np.arange(0,self.n),np.arange(0,self.n+1)
        if self.q == 0:
            prod = np.power(np.nanmin(np.divide(q_available, self.j_a[srv_idx,:][:,srv_idx_a]),
                                      axis=1), self.b)
            # print(prod)
            return prod
        elif self.q == np.inf:
            return np.power(np.nanprod(np.power(np.divide(q_available, self.j_a[srv_idx,:][:,srv_idx_a]),
                                                self.a_a[srv_idx,:][:,srv_idx_a]),
                                       axis=1),
                            self.b)
        else:
            return np.power(np.nansum(self.a_a[srv_idx,:][:,srv_idx_a] * np.power(self.j_a[srv_idx,:][:,srv_idx_a], 1. / self.q)
                                      / np.power(q_available, 1. / self.q), axis=1),
                            - self.b * self.q)
    
    def get_alone_firms_firms(self):
        alone_firms = np.where((self.j.sum(axis = 1)==0)&(self.j.sum(axis = 0)==0))[0]
        return alone_firms
    
    def clean_network(self,bkrpt_idx, srv_idx):
        alone_firms = self.get_alone_firms_firms()
        if self.q == 0.:
            firms_no_production = np.where(self.leontief_net.sum(axis = 1)  - (self.j*self.leontief_net).sum(axis = 1) != 0)[0]
            alone_firms = np.unique(np.concatenate([alone_firms,firms_no_production]))
        bkrpt_idx_ = np.unique(np.concatenate([alone_firms,bkrpt_idx]))
        while set(bkrpt_idx_) - set(bkrpt_idx):
            bkrpt_idx = np.unique(np.concatenate([bkrpt_idx,bkrpt_idx_]))
            self.j[:,bkrpt_idx] = 0
            self.j[bkrpt_idx,:] = 0
            self.j0[bkrpt_idx]  = 0
            alone_firms = self.get_alone_firms_firms()
            if self.q == 0.:
                firms_no_production = np.where(self.leontief_net.sum(axis = 1)  - (self.j*self.leontief_net).sum(axis = 1) != 0)[0]
                alone_firms = np.unique(np.concatenate([alone_firms,firms_no_production]))
            bkrpt_idx_ = np.unique(np.concatenate([alone_firms,bkrpt_idx]))
        bkrpt_idx = bkrpt_idx_
        return bkrpt_idx, np.setxor1d(self.list_firms,bkrpt_idx)

    def update_j(self,bkrpt_idx,srv_idx):
        """Author : Swann Chelly
        Update_j when profits are lower than a threshold -min_loss. Destroy all links of a firm when it goes bankrupt. 
        Args:
            min_loss (float): Threshold under which firms go bankrupt
            gains (np.array): gains table
            losses (np.array): gains table
            interest_rates (np.array, optional): Compute cumulated profits according to interest rates. Defaults to None.
        """
        # plt.matshow(self.j)
        t1 = time()
        self.j[:,bkrpt_idx] = 0
        self.j[bkrpt_idx,:] = 0
        self.j0[bkrpt_idx]  = 0
        
        bkrpt_idx,srv_idx = self.clean_network(bkrpt_idx,srv_idx)

        if len(bkrpt_idx)==self.n:
            return bkrpt_idx,srv_idx
        
        redist,weight = np.ones(self.n).reshape(-1,1),np.ones(self.n).reshape(-1,1)
        redist[srv_idx] = self.a_a[srv_idx,:][:,bkrpt_idx+1].sum(axis = 1).reshape(-1,1)
        weight[srv_idx] = self.a_a[srv_idx,:][:,srv_idx+1].sum(axis = 1).reshape(-1,1)
        # redist[weight == 0] = 0
        weight[weight == 0] = 1 # To avoid collapse

        self.a_a[:,bkrpt_idx+1] = 0
        self.a_a[bkrpt_idx,:]   = 0
        self.a_a[:,1:]  = self.a_a[:,1:]*(1+redist/weight)

        labour_only = np.where(self.a_a[srv_idx,:][:,1:].sum(axis = 1) == 0)
        self.a_a[srv_idx[labour_only],0] = 1. 
        
        # Sanity check, all sum of rows of self.a_a must be either 0 or 1.
        temp_ = set(np.round(self.a_a.sum(axis = 1),6))
        assert temp_ == set([0,1]), f"Sum of a_a contains :{temp_-set([0,1])} {set(srv_idx)}" 
        assert np.isnan(self.a_a).sum() == 0 , "NaN in a_a"

        self.house.theta[bkrpt_idx] = 0
        self.house.theta[srv_idx] = self.house.theta[srv_idx]/self.house.theta[srv_idx].sum()
        # self.firms.z[bkrpt_idx] = 0
        self.a0,self.a = self.a_a[:,0],self.a_a[:,1:]
        self.set_quantities()
        self.time_table_1.append(time()-t1)
        t1 = time()
        self.compute_eq(srv_idx)
        self.time_table_2.append(time()-t1)
        return bkrpt_idx,srv_idx

    def compute_eq(self,srv_idx = None):
        """
        Computes the competitive equilibrium of the economy. We use least-squares to compute solutions of linear
        systems Ax=b for memory and computational efficiency. The non-linear equations for non-constant return to scale
        parameters are solved using generalized least-squares with initial guesses taken to be the solution of the b=1
        linear equation. For a high number of firms, high heterogeneity of close to 0 epsilon, this function might
        can output erroneous results or errors.
        :return: side effect.
        """
        if srv_idx is None : srv_idx,srv_idx_a = np.arange(0,self.n),np.arange(0,self.n+1)
        else : srv_idx_a = np.concatenate([[0],srv_idx+1])
        n_ = len(srv_idx)
        self.p_eq,self.g_eq,self.cons_eq = np.zeros((self.n)),np.zeros((self.n)),np.zeros((self.n))
        
        if self.q == np.inf:
            h = np.sum(
                self.a_a[srv_idx,:][:,srv_idx_a] * np.log(np.ma.masked_invalid(np.divide(self.j_a[srv_idx,:][:,srv_idx_a], self.a_a[srv_idx,:][:,srv_idx_a]))), axis=1)
            v = lstsq(np.eye(n_) - self.a[srv_idx,:][:,srv_idx].T,
                      self.kappa[srv_idx],
                      rcond=10e-7)[0]
            log_p = lstsq(np.eye(n_) / self.b - self.a[srv_idx,:][:,srv_idx],
                          - np.log(self.firms.z[srv_idx]) / self.b +
                          (1 - self.b) * np.log(v) / self.b + h,
                          rcond=10e-7)[0]
            log_g = - np.log(self.firms.z[srv_idx]) - log_p + np.log(v)
            self.p_eq[srv_idx], self.g_eq[srv_idx] = np.exp(log_p), np.exp(log_g)
        else:
            if self.b != 1:
                if self.q == 0:
                    init_guess_peq = lstsq(self.m_cal[srv_idx,:][:,srv_idx],
                                           self.v[srv_idx],
                                           rcond=10e-7)[0]
                    init_guess_geq = lstsq(self.m_cal[srv_idx,:][:,srv_idx].T,
                                           np.divide(
                                               self.kappa[srv_idx], init_guess_peq),
                                           rcond=10e-7)[0]

                    par = (self.firms.z[srv_idx],
                           self.v[srv_idx],
                           self.m_cal[srv_idx,:][:,srv_idx],
                           self.b - 1,
                           self.kappa[srv_idx])

                    pert_peq = lstsq(
                        self.m_cal[srv_idx,:][:,srv_idx], self.firms.z[srv_idx] * init_guess_peq * np.log(init_guess_geq), rcond=10e-7)[0]

                    pert_geq = lstsq(np.transpose(self.m_cal[srv_idx,:][:,srv_idx]),
                                     - np.divide(self.kappa[srv_idx],
                                                 np.power(init_guess_peq, 2)) * pert_peq +
                                     self.firms.z[srv_idx] * init_guess_geq *
                                     np.log(init_guess_geq),
                                     rcond=10e-7)[0]

                    pg = leastsq(lambda x: self.non_linear_eq_qzero(x, *par),
                                 np.array(np.concatenate((init_guess_peq + (1 - self.b) * pert_peq,
                                                          np.power(init_guess_geq + (1 - self.b) * (
                                                              pert_geq - init_guess_geq * np.log(init_guess_geq)),
                                     1 / self.b))).reshape(2 * n_)))[0]

                    # pylint: disable=unbalanced-tuple-unpacking
                    self.p_eq[srv_idx], g = np.split(pg, 2)
                    self.g_eq[srv_idx] = np.power(g, self.b)

                else:

                    # The numerical solving is done for variables u = p_eq ^ zeta and
                    # w = z ^ (q * zeta) * u ^ q * g_eq ^ (zeta * (bq+1) / b)

                    init_guess_u = lstsq(self.m_cal[srv_idx,:][:,srv_idx],
                                         self.v[srv_idx],
                                         rcond=None)[0]
                    init_guess_w = lstsq(self.m_cal[srv_idx,:][:,srv_idx].T,
                                         np.divide(self.kappa[srv_idx], init_guess_u),
                                         rcond=None)[0]

                    par = (np.power(self.firms.z[srv_idx], self.zeta),
                           self.v[srv_idx],
                           self.m_cal[srv_idx,:][:,srv_idx],
                           self.q,
                           (self.b - 1) / (self.b * self.q + 1),
                           self.kappa[srv_idx]
                           )

                    uw = leastsq(lambda x: self.non_linear_eq_qnonzero(x, *par),
                                 np.concatenate((init_guess_u, init_guess_w)),
                                 )[0]

                    # pylint: disable=unbalanced-tuple-unpacking
                    u, w = np.split(uw, 2)
                    self.p_eq[srv_idx] = np.power(u, 1. / self.zeta)
                    self.g_eq[srv_idx] = np.power(np.divide(w, np.power(self.firms.z[srv_idx], self.q * self.zeta) * np.power(u, self.q)),
                                         self.b / (self.zeta * (self.b * self.q + 1)))
            else:
                if self.q == 0:
                    self.p_eq[srv_idx] = lstsq(self.m_cal[srv_idx,:][:,srv_idx],
                                      self.v[srv_idx],
                                      rcond=10e-7)[0]
                    self.g_eq[srv_idx] = lstsq(self.m_cal[srv_idx,:][:,srv_idx].T,
                                      np.divide(self.kappa[srv_idx], self.p_eq[srv_idx]),
                                      rcond=10e-7)[0]
                else:

                    # The numerical solving is done for variables u = p_eq ^ zeta and
                    # w = z ^ (q * zeta) * u ^ q * g_eq

                    u = lstsq(self.m_cal[srv_idx,:][:,srv_idx],
                              self.v[srv_idx],
                              rcond=None)[0]
                    self.p_eq[srv_idx] = np.power(u, 1. / self.zeta)
                    w = lstsq(self.m_cal[srv_idx,:][:,srv_idx].T,
                              np.divide(self.kappa[srv_idx], u),
                              rcond=None)[0]
                    self.g_eq[srv_idx] = np.divide(w, np.power(
                        self.firms.z[srv_idx], self.q * self.zeta) * np.power(u, self.q))

        self.labour_eq = np.power(
            self.mu_eq * self.house.f, 1. / self.house.phi) / self.house.v_phi
        self.cons_eq[srv_idx] = self.kappa[srv_idx] / self.p_eq[srv_idx]
        self.b_eq = np.sum(self.house.theta[srv_idx]) / self.mu_eq
        self.utility_eq = np.dot(self.house.theta[srv_idx], np.log(self.cons_eq[srv_idx])) - self.house.gamma * np.power(
            self.labour_eq / self.house.l_0,
            self.house.phi + 1) / (
                self.house.phi + 1)

    def save_eco(self, name):
        """
        Saves the economy as multi-indexed data-frame in hdf format along with networks in npy format.
        :param name: name of file,
        """
        first_index = np.concatenate(
            (np.repeat('Firms', 11), np.repeat('Household', 11)))
        second_index = np.concatenate(
            (['q', 'b', 'z', 'sigma', 'alpha', 'alpha_p', 'beta', 'beta_p', 'w', 'p_eq', 'g_eq'],
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
                            self.house.l_0 * np.ones(self.n),
                            self.house.theta,
                            self.house.gamma * np.ones(self.n),
                            self.house.phi * np.ones(self.n),
                            ))

        df_eco = pd.DataFrame(values,
                              index=multi_index,
                              columns=[np.arange(1, self.n + 1)]
                              )
        df_eco.to_hdf(name + '/eco.h5', key='df', mode='w')
        np.save(name + '/network.npy', self.j_a)
        if self.q != 0:
            np.save(name + '/sub_network.npy', self.a_a)

    # Fixed point equations for equilibrium computation

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
        u, w = np.split(x, 2)
        z_zeta, v, m_cal, q, exponent, kappa = p
        w_over_uq_p = np.power(
            np.divide(w, np.power(z_zeta, q) * np.power(u, q)), exponent)
        v1 = np.multiply(z_zeta, np.multiply(u, 1 - w_over_uq_p))
        m1 = np.dot(m_cal, u)
        m2 = u * np.dot(m_cal.T, w) - w * m1
        return np.concatenate((m1 - v1 - v, m2 + w * v - kappa))

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
