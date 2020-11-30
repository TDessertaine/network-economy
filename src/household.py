# network-economy is a simulation program for the Network Economy ABM desbribed in (TODO)
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
The ``household`` module
======================

This module declares the Household class which model one representative household.
The attributes of this class are all the fixed parameters defining the household.
"""
import numpy as np


class Household(object):

    def __init__(self, l_0, theta, gamma, phi, omega_p=None):
        # Primary instances
        self.l_0 = l_0
        self.theta = theta
        self.gamma = gamma
        self.phi = phi
        self.omega_p = omega_p if omega_p else 0

        # Secondary instances
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l_0, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(np.sum(self.theta) * self.v_phi,
                                           self.phi / (1 + self.phi))

    # Setters for class instances

    def update_labour(self, labour):
        self.l_0 = labour
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(labour, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(np.sum(self.theta) * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_theta(self, theta):
        self.theta = theta
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l_0, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(np.sum(self.theta) * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_gamma(self, gamma):
        self.gamma = gamma
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l_0, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(np.sum(self.theta) * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_phi(self, phi):
        self.phi = phi
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l_0, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(np.sum(self.theta) * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_w_p(self, omega_p):
        self.omega_p = omega_p

    def utility(self, consumption, working_hours):
        """
        Utility function.
        :param consumption: current consumption,
        :param working_hours: realized labor hours,
        :return: Value of the utility function.
        """
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(working_hours.sum() / self.l_0,
                                                                                1. + self.phi) / (
                       1. + self.phi)

    def compute_demand_cons_labour_supply(self, savings, prices, labour_supply, labour_demand, step_s, f):
        """
        Optimization sequence carried by the household.
        :param savings: wage-rescaled savings for the next period,
        :param prices: wage-rescaled prices for the next period,
        :param labour_supply: realized supply of labor of the current period,
        :param labour_demand: realized demand for labor of the current period,
        :param step_s: size of time-step,
        :return: Consumption targets and labor supply for the next period.
        """

        # Update preferences taking confidence effects into account
        theta = self.theta * np.exp(- self.omega_p * step_s * (labour_supply - labour_demand) /
                                    (labour_supply + labour_demand))

        if self.phi == 1:
            mu = .5 * (np.sqrt(np.power(savings * self.v_phi, 2)
                               + 4 * self.v_phi * np.sum(theta) * f**2)
                       - savings * self.v_phi) / f**2
        elif self.phi == np.inf:
            mu = np.sum(theta) / (self.l_0 * f + savings)
        else:
            # (TODO)
            raise Exception('Not coded yet')
            # x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            # mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))

        return theta / (mu * prices), np.power(mu * f, 1. / self.phi) / self.v_phi

    @staticmethod
    def fixed_point_mu(x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x
