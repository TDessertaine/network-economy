# network-economy is a simulation program for the Network Economy ABM desbribed
# in <https://doi.org/10.1016/j.jedc.2022.104362>
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
This class has the network attributes (both input-output and substitution)
along with subsequent quantities (equilibrium etc). It also inherits firms and
households attributes.
"""
import warnings

import numpy as np
import pandas as pd

from numpy.linalg import lstsq
from scipy.optimize import leastsq

import errors_strings
from firms import Firms
from household import Household
from network import create_net

warnings.simplefilter("ignore")
CODED_NETWORK_TYPE = ["regular", "m-regular", "er"]

# pylint: disable=too-many-arguments


class Economy:
    def __init__(
        self,
        firms_number: int = None,
        average_connectivity: int = None,
        network_type: str = None,
        is_directed: bool = None,
        work_vector: np.array = None,
        work_substitution_vector: np.array = None,
        ces_parameter: float = None,
        returns_to_scale: float = None,
    ) -> None:
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

        if firms_number is not None and firms_number <= 0:
            raise ValueError("n must be a positive integer.")

        if average_connectivity is not None and average_connectivity < 0:
            raise ValueError("d must be a positive integer.")

        if network_type is not None and network_type not in CODED_NETWORK_TYPE:
            raise ValueError(
                "netstring not supported. Please choose 'regular' for regular, 'm-regular'"
                + " for multi-regular, 'er' for Erdös-Renyi."
            )

        if work_vector is not None and (work_vector < 0).any():
            raise ValueError("Entries of j0 must be greater or equal to 0.")

        if (
            work_substitution_vector is not None
            and (work_substitution_vector < 0).any()
            or (work_substitution_vector > 1).any()
        ):
            raise ValueError("Entries of a0 must be between 0 and 1.")

        if ces_parameter is not None and ces_parameter < 0:
            raise ValueError("q must be a positive real number.")

        if returns_to_scale is not None and returns_to_scale < 0:
            raise ValueError("b must be a positive real number.")

        # Network initialization
        self.firms_number = firms_number if firms_number else 100
        average_connectivity = average_connectivity if average_connectivity else 15
        network_type = network_type if network_type else "m_regular"
        is_directed = is_directed if is_directed else True

        self.adjacency_network = create_net(
            network_type, is_directed, firms_number, average_connectivity
        )  # Network creation
        self.work_vector = work_vector if work_vector else np.ones(self.firms_number)
        self.work_substitution_vector = (
            work_substitution_vector
            if work_substitution_vector
            else 0.5 * np.ones(self.firms_number)
        )

        substitution_matrix_without_work = np.multiply(
            np.random.uniform(0, 1, (self.firms_number, self.firms_number)),
            self.adjacency_network,
        )

        self.substitution_matrix = np.array(
            [
                (1 - work_substitution_vector[i])
                * substitution_matrix_without_work[i]
                / np.sum(substitution_matrix_without_work[i])
                for i in range(firms_number)
            ]
        )

        self.augmented_substitution_matrix = np.hstack(
            (np.array([self.work_substitution_vector]).T, self.substitution_matrix)
        )
        self.augmented_adjacency_matrix = np.hstack(
            (np.array([self.work_vector]).T, self.adjacency_network)
        )

        self.ces_parameter = ces_parameter if ces_parameter else 0
        self.auxiliary_ces_parameter = 1 / (self.ces_parameter + 1)
        self.returns_to_scale = returns_to_scale if returns_to_scale else 1

        # Auxiliary network variables
        self.aggregate_matrix = None

        self.augmented_aggregate_matrix = None
        self.network_matrix = None
        self.household_consumption_theory = None
        self.zeros_augmented_network = None

        # Firms and household sub-classes
        self.firms_sector = None
        self.household_sector = None

        # Equilibrium quantities
        self.equilibrium_prices = None
        self.equilibrium_production_levels = None
        self.equilibrium_budget_checker = None
        self.equilibrium_labour = None
        self.equilibrium_consumptions = None
        self.equilibrium_budget = None
        self.equilibrium_utility = None

    def init_house(
        self,
        labour_baseline: float = None,
        preferency_factors: np.array = None,
        work_desutility_paramater: float = None,
        frisch_index: float = None,
        omega_prime: float = None,
        fraction_to_consume: float = None,
        interest_rate: float = None,
    ) -> None:

        r"""Initialize a Household object as an instance of the Economy object.
        Refer to initialization of Household class.

        Args:
            labour_baseline (float, optional):
                Baseline work offer, corresponds to $L_0$.
                Must be positive.
                Default to 1.
            preferency_factors (np.array, optional):
                Vector of preferency factors, corresponds to $\theta$.
                Must be a positive array.
                Default to np.ones(firms_number)/firms_number.
            work_desutility_paramater (float, optional):
                Aversion to work parameter, corresponds to $\Gamma$.
                Must be positive.
                Default to 1.
            frisch_index (float, optional):
                Frisch index, corresponds to $\phi$.
                Must be positive.
                Default to 1.
            omega_prime (float, optional):
                Adjustment speed of consumption's confidence, corresponds to $\omega^\prime$.
                Must be positive.
                Default to 0.
            fraction_to_consume (float, optional):
                Fraction of the budget allocated to consumption, corresponds to $f$.
                Must be between 0 and 1.
                Defaults to 1.
            interest_rate (float, optional):
                Interest rate on savings, corresponds to $r$.
                Must be between 0 and 1.
                Defaults to 0.
        """

        if labour_baseline is not None and labour_baseline <= 0:
            raise ValueError("labour_basline must be positive.")

        if preferency_factors is not None and (preferency_factors < 0).any():
            raise ValueError("Entries of preferency_factor must be positive.")

        if work_desutility_paramater is not None and work_desutility_paramater < 0:
            raise ValueError("work_desutility_paramater must be positive.")

        if frisch_index is not None and frisch_index < 0:
            raise ValueError("frisch_index must be positive.")

        if omega_prime is not None and omega_prime < 0:
            raise ValueError("omega_prime must be positive.")

        if fraction_to_consume is not None and (
            fraction_to_consume > 1 or fraction_to_consume < 0
        ):
            raise ValueError("fraction_to_consume must be between 0 and 1.")

        if interest_rate is not None and (interest_rate > 1 or interest_rate < 0):
            raise ValueError("interest_rate must be between 0 and 1.")

        labour_baseline = labour_baseline if labour_baseline else 1.0
        preferency_factors = (
            preferency_factors
            if preferency_factors
            else np.ones(self.firms_number) / self.firms_number
        )
        work_desutility_paramater = (
            work_desutility_paramater if work_desutility_paramater else 1.0
        )
        frisch_index = frisch_index if frisch_index else 1.0
        omega_prime = omega_prime if omega_prime else 0.0
        fraction_to_consume = fraction_to_consume if fraction_to_consume else 1.0
        interest_rate = interest_rate if interest_rate else 0.0

        self.household_sector = Household(
            labour_baseline,
            preferency_factors,
            work_desutility_paramater,
            frisch_index,
            omega_prime,
            fraction_to_consume,
            interest_rate,
        )

    def init_firms(
        self,
        productivity_factors: np.array = None,
        depreciation_stock: np.array = None,
        price_surplus_coupling: float = None,
        price_profit_coupling: float = None,
        production_profit_coupling: float = None,
        production_surplus_coupling: float = None,
        wage_labour_coupling: float = None,
    ) -> None:
        """Initializes a Firms object as an instance of the Economy object.
        Refer to initialization of Firms class.

        Args:
            productivity_factors (np.array):
                Vector of productivity factors, corresponds to $z$.
                Must be positive.
                Defaults to np.ones(self.firms_number).
            depreciation_stock (np.array):
                Depreciation of stocks parameters, corresponds to $\sigma$.
                Must be positive.
                Defaults to np.ones(self.firms_number).
            price_surplus_coupling (float):
                Adjustement speed of prices with respect to surplus, corresponds to \u03B1.
                Must be positive.
                Defaults to 0.25.
            price_profit_coupling (float):
                Adjustement speed of prices with respect to profits, corresponds to $\alpha^\prime$.
                Must be positive.
                Defaults to 0.25.
            production_profit_coupling (float):
                Adjustement speed of productions with respect to profits, corresponds to $\beta$.
                Must be positive.
                Defaults to 0.25.
            production_surplus_coupling (float):
                Adjustement speed of productions with respect to surplus, corresponds to $\beta^\prime$.
                Must be positive.
                Defaults to 0.25.
            wage_labour_coupling (float):
                Adjustement speed of wages with respect to labour market tensions, corresponds to $\omega$.
                Must be positive.
                Defaults to 0.1.
        """

        self.firms_sector = Firms(
            productivity_factors,
            depreciation_stock,
            price_surplus_coupling,
            price_profit_coupling,
            production_profit_coupling,
            production_surplus_coupling,
            wage_labour_coupling,
        )

    # Setters for class instances

    def set_house(self, household: Household = None) -> None:
        """Sets an instanciated Household object as the household_sector instance
        of the Economy object.

        Args:
            house (Household): an instanciated Household object. Defaults to None.
        """

        if household is not None:
            self.household_sector = household
        else:
            print(
                "Provided Household object was None. Instanciated with default instead."
            )
            self.init_house()

    def set_firms(self, firms: Firms = None) -> None:
        """Sets an instanciated Firms object as the household_sector instance of the Economy object.

        Args:
            firms (Firms): an instanciated Firms object. Defaults to None.
        """
        if firms is not None:
            self.firms_sector = firms
        else:
            print("Provided Firms object was None. Instanciated with default instead.")
            self.init_firms()

    # Update methods for firms and household

    def update_firms_productivity_factors(
        self, productivity_factors: np.array = None
    ) -> None:
        """Updates productivity factors of firms_sector instance.

        Args:
            productivity_factors (np.array): a vector of productivity factors with positive entries.
            Defaults to np.ones(self.firms_number).
        """

        if productivity_factors is not None:
            if (productivity_factors < 0).any():
                raise ValueError("Entries of productivity_factors must be positive.")
            self.firms_sector.update_productivity_factors(productivity_factors)
        else:
            print(
                "Provided productivity_factors object was None. Instanciated with default instead."
            )
            self.firms_sector.update_productivity_factors(np.ones(self.firms_number))
        self.set_quantities()
        self.compute_eq()

    def update_firms_sigma(self, sigma):
        self.firms_sector.update_sigma(sigma)

    def update_firms_alpha(self, alpha):
        self.firms_sector.update_alpha(alpha)

    def update_firms_alpha_p(self, alpha_p):
        self.firms_sector.update_alpha_p(alpha_p)

    def update_firms_beta(self, beta):
        self.firms_sector.update_beta(beta)

    def update_firms_beta_p(self, beta_p):
        self.firms_sector.update_beta_p(beta_p)

    def update_firms_w(self, omega):
        self.firms_sector.update_w(omega)

    def update_house_labour(self, labour):
        self.household_sector.update_labour(labour)
        self.compute_eq()

    def update_house_theta(self, theta):
        self.household_sector.update_theta(theta)
        self.compute_eq()

    def update_house_gamma(self, gamma):
        self.household_sector.update_gamma(gamma)
        self.compute_eq()

    def update_house_phi(self, phi):
        self.household_sector.update_phi(phi)
        self.compute_eq()

    def update_house_w_p(self, omega_p):
        self.household_sector.update_w_p(omega_p)

    def update_house_f(self, f):
        self.household_sector.update_f(f)

    def update_house_r(self, r):
        self.household_sector.update_r(r)

    # Setters for the networks and subsequent instances

    def set_j(self, j):
        """
        Sets a particular input-output network.
        :param j: a n by n matrix.
        :return: side effect
        """
        if j.shape != (self.firms_number, self.firms_number):
            raise ValueError(
                "Input-output network must be of size (%d, %d)"
                % (self.firms_number, self.firms_number)
            )

        self.adjacency_network = j
        self.set_quantities()
        self.compute_eq()

    def set_a(self, a):
        """
        Sets a particular input-output network.
        :param a: a n by n matrix.
        :return: side effect
        """
        if a.shape != (self.firms_number, self.firms_number):
            raise ValueError(
                "Substitution network must be of size (%d, %d)"
                % (self.firms_number, self.firms_number)
            )
        self.substitution_matrix = a
        self.set_quantities()
        self.compute_eq()

    def set_quantities(self):
        """
        Sets redundant economy quantities as class instances.
        :return: side effect
        """
        if self.ces_parameter == 0:
            self.aggregate_matrix = self.adjacency_network
            self.augmented_aggregate_matrix = self.augmented_adjacency_matrix
            self.network_matrix = (
                np.diag(self.firms_sector.productivity_factors) - self.aggregate_matrix
            )
        elif self.ces_parameter == np.inf:
            self.aggregate_matrix = self.substitution_matrix
            self.augmented_aggregate_matrix = self.augmented_substitution_matrix
            self.network_matrix = np.eye(self.firms_number) - self.aggregate_matrix
        else:
            self.aggregate_matrix = np.multiply(
                np.power(
                    self.substitution_matrix,
                    self.ces_parameter * self.auxiliary_ces_parameter,
                ),
                np.power(self.adjacency_network, self.auxiliary_ces_parameter),
            )
            self.augmented_aggregate_matrix = np.multiply(
                np.power(
                    self.augmented_substitution_matrix,
                    self.ces_parameter * self.auxiliary_ces_parameter,
                ),
                np.power(self.augmented_adjacency_matrix, self.auxiliary_ces_parameter),
            )
            self.network_matrix = (
                np.diag(
                    np.power(
                        self.firms_sector.productivity_factors,
                        self.auxiliary_ces_parameter,
                    )
                )
                - self.aggregate_matrix
            )
        self.zeros_augmented_network = self.augmented_adjacency_matrix != 0

    def get_eps_cal(self):
        """
        Computes the smallest eigenvalue of the economy matrix
        :return: smallest eigenvalue
        """
        return np.min(np.real(np.linalg.eigvals(self.network_matrix)))

    def set_eps_cal(self, eps):
        """
        Modifies firms instance to set smallest eigenvalue of economy matrix to given epsilon.
        :param eps: a real number,
        :return: side effect.
        """
        min_eig = self.get_eps_cal()
        z_n = self.firms_sector.productivity_factors * np.power(
            1
            + (eps - min_eig)
            / np.power(
                self.firms_sector.productivity_factors, self.auxiliary_ces_parameter
            ),
            self.ces_parameter + 1,
        )
        sigma = self.firms_sector.depreciation_stock
        alpha = self.firms_sector.price_surplus_coupling
        alpha_p = self.firms_sector.price_profit_coupling
        beta = self.firms_sector.production_profit_coupling
        beta_p = self.firms_sector.production_surplus_coupling
        omega = self.firms_sector.wage_labour_coupling
        self.init_firms(z_n, sigma, alpha, alpha_p, beta, beta_p, omega)
        self.set_quantities()
        self.compute_eq()

    def update_b(self, b):
        """
        Sets return to scale parameter
        :param b: return to scale
        :return: side effect
        """
        self.returns_to_scale = b
        self.compute_eq()

    def update_q(self, q):
        self.ces_parameter = q
        self.auxiliary_ces_parameter = 1 / (q + 1)
        self.set_quantities()
        self.compute_eq()

    def update_network(self, netstring, directed, d, n):
        self.adjacency_network = create_net(netstring, directed, n, d)
        a = np.multiply(np.random.uniform(0, 1, (n, n)), self.adjacency_network)
        self.substitution_matrix = np.array(
            [
                (1 - self.work_substitution_vector[i]) * a[i] / np.sum(a[i])
                for i in range(n)
            ]
        )
        self.set_quantities()
        self.compute_eq()

    def update_a0(self, a0):
        self.work_substitution_vector = a0
        a = np.multiply(
            np.random.uniform(0, 1, (self.firms_number, self.firms_number)),
            self.adjacency_network,
        )
        self.substitution_matrix = np.array(
            [
                (1 - self.work_substitution_vector[i]) * a[i] / np.sum(a[i])
                for i in range(self.firms_number)
            ]
        )
        self.set_quantities()
        self.compute_eq()

    def update_j0(self, j0):
        self.work_vector = j0
        self.set_quantities()
        self.compute_eq()

    def production_function(self, q_available):
        """
        CES production function.
        :param q_available: matrix of available labour and goods for production,
        :return: production levels of the firms.
        """
        if self.ces_parameter == 0:
            prod = np.power(
                np.nanmin(
                    np.divide(q_available, self.augmented_adjacency_matrix), axis=1
                ),
                self.returns_to_scale,
            )
            # print(prod)
            return prod
        elif self.ces_parameter == np.inf:
            return np.power(
                np.nanprod(
                    np.power(
                        np.divide(q_available, self.augmented_adjacency_matrix),
                        self.augmented_substitution_matrix,
                    ),
                    axis=1,
                ),
                self.returns_to_scale,
            )
        else:
            return np.power(
                np.nansum(
                    self.augmented_substitution_matrix
                    * np.power(
                        self.augmented_adjacency_matrix, 1.0 / self.ces_parameter
                    )
                    / np.power(q_available, 1.0 / self.ces_parameter),
                    axis=1,
                ),
                -self.returns_to_scale * self.ces_parameter,
            )

    def compute_eq(self) -> None:
        """
        Computes the competitive equilibrium of the economy. We use least-squares to compute solutions of linear
        systems Ax=b for memory and computational efficiency. The non-linear equations for non-constant return to scale
        parameters are solved using generalized least-squares with initial guesses taken to be the solution of the b=1
        linear equation. For a high number of firms, high heterogeneity of close to 0 epsilon, this function might
        can output erroneous results or errors.
        :return: side effect.
        """

        self.equilibrium_budget_checker = np.power(
            np.power(self.household_sector.gamma, 1.0 / self.household_sector.phi)
            * np.sum(self.household_sector.theta)
            * (1 - (1 - self.household_sector.f) * (1 + self.household_sector.r))
            / (
                self.household_sector.f
                * np.power(
                    self.household_sector.l_0, 1 + 1.0 / self.household_sector.phi
                )
            ),
            self.household_sector.phi / (1 + self.household_sector.phi),
        )
        self.household_consumption_theory = (
            self.household_sector.theta / self.equilibrium_budget_checker
        )

        if self.ces_parameter == np.inf:
            h = np.sum(
                self.augmented_substitution_matrix
                * np.log(
                    np.ma.masked_invalid(
                        np.divide(
                            self.augmented_adjacency_matrix,
                            self.augmented_substitution_matrix,
                        )
                    )
                ),
                axis=1,
            )
            v = lstsq(
                np.eye(self.firms_number) - self.substitution_matrix.T,
                self.household_consumption_theory,
                rcond=10e-7,
            )[0]
            log_p = lstsq(
                np.eye(self.firms_number) / self.returns_to_scale
                - self.substitution_matrix,
                -np.log(self.firms_sector.productivity_factors) / self.returns_to_scale
                + (1 - self.returns_to_scale) * np.log(v) / self.returns_to_scale
                + h,
                rcond=10e-7,
            )[0]
            log_g = -np.log(self.firms_sector.productivity_factors) - log_p + np.log(v)
            self.equilibrium_prices, self.equilibrium_production_levels = np.exp(
                log_p
            ), np.exp(log_g)
        else:
            if self.returns_to_scale != 1:
                if self.ces_parameter == 0:
                    init_guess_peq = lstsq(
                        self.network_matrix,
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        rcond=10e-7,
                    )[0]
                    init_guess_geq = lstsq(
                        self.network_matrix.T,
                        np.divide(self.household_consumption_theory, init_guess_peq),
                        rcond=10e-7,
                    )[0]

                    par = (
                        self.firms_sector.productivity_factors,
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        self.network_matrix,
                        self.returns_to_scale - 1,
                        self.household_consumption_theory,
                    )

                    pert_peq = lstsq(
                        self.network_matrix,
                        self.firms_sector.productivity_factors
                        * init_guess_peq
                        * np.log(init_guess_geq),
                        rcond=10e-7,
                    )[0]

                    pert_geq = lstsq(
                        np.transpose(self.network_matrix),
                        -np.divide(
                            self.household_consumption_theory,
                            np.power(init_guess_peq, 2),
                        )
                        * pert_peq
                        + self.firms_sector.productivity_factors
                        * init_guess_geq
                        * np.log(init_guess_geq),
                        rcond=10e-7,
                    )[0]

                    pg = leastsq(
                        lambda x: self.non_linear_eq_qzero(x, *par),
                        np.array(
                            np.concatenate(
                                (
                                    init_guess_peq
                                    + (1 - self.returns_to_scale) * pert_peq,
                                    np.power(
                                        init_guess_geq
                                        + (1 - self.returns_to_scale)
                                        * (
                                            pert_geq
                                            - init_guess_geq * np.log(init_guess_geq)
                                        ),
                                        1 / self.returns_to_scale,
                                    ),
                                )
                            ).reshape(2 * self.firms_number)
                        ),
                    )[0]

                    # pylint: disable=unbalanced-tuple-unpacking
                    self.equilibrium_prices, g = np.split(pg, 2)
                    self.equilibrium_production_levels = np.power(
                        g, self.returns_to_scale
                    )

                else:

                    # The numerical solving is done for variables u = p_eq ^ zeta and
                    # w = z ^ (q * zeta) * u ^ q * g_eq ^ (zeta * (bq+1) / b)

                    init_guess_u = lstsq(
                        self.network_matrix,
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        rcond=None,
                    )[0]
                    init_guess_w = lstsq(
                        self.network_matrix.T,
                        np.divide(self.household_consumption_theory, init_guess_u),
                        rcond=None,
                    )[0]

                    par = (
                        np.power(
                            self.firms_sector.productivity_factors,
                            self.auxiliary_ces_parameter,
                        ),
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        self.network_matrix,
                        self.ces_parameter,
                        (self.returns_to_scale - 1)
                        / (self.returns_to_scale * self.ces_parameter + 1),
                        self.household_consumption_theory,
                    )

                    uw = leastsq(
                        lambda x: self.non_linear_eq_qnonzero(x, *par),
                        np.concatenate((init_guess_u, init_guess_w)),
                    )[0]

                    # pylint: disable=unbalanced-tuple-unpacking
                    u, w = np.split(uw, 2)
                    self.equilibrium_prices = np.power(
                        u, 1.0 / self.auxiliary_ces_parameter
                    )
                    self.equilibrium_production_levels = np.power(
                        np.divide(
                            w,
                            np.power(
                                self.firms_sector.productivity_factors,
                                self.ces_parameter * self.auxiliary_ces_parameter,
                            )
                            * np.power(u, self.ces_parameter),
                        ),
                        self.returns_to_scale
                        / (
                            self.auxiliary_ces_parameter
                            * (self.returns_to_scale * self.ces_parameter + 1)
                        ),
                    )
            else:
                if self.ces_parameter == 0:
                    self.equilibrium_prices = lstsq(
                        self.network_matrix,
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        rcond=10e-7,
                    )[0]
                    self.equilibrium_production_levels = lstsq(
                        self.network_matrix.T,
                        np.divide(
                            self.household_consumption_theory, self.equilibrium_prices
                        ),
                        rcond=10e-7,
                    )[0]
                else:

                    # The numerical solving is done for variables u = p_eq ^ zeta and
                    # w = z ^ (q * zeta) * u ^ q * g_eq

                    u = lstsq(
                        self.network_matrix,
                        np.array(self.augmented_aggregate_matrix[:, 0]),
                        rcond=None,
                    )[0]
                    self.equilibrium_prices = np.power(
                        u, 1.0 / self.auxiliary_ces_parameter
                    )
                    w = lstsq(
                        self.network_matrix.T,
                        np.divide(self.household_consumption_theory, u),
                        rcond=None,
                    )[0]
                    self.equilibrium_production_levels = np.divide(
                        w,
                        np.power(
                            self.firms_sector.productivity_factors,
                            self.ces_parameter * self.auxiliary_ces_parameter,
                        )
                        * np.power(u, self.ces_parameter),
                    )

        self.equilibrium_labour = (
            np.power(
                self.equilibrium_budget_checker * self.household_sector.f,
                1.0 / self.household_sector.phi,
            )
            / self.household_sector.v_phi
        )
        self.equilibrium_consumptions = (
            self.household_consumption_theory / self.equilibrium_prices
        )
        self.equilibrium_budget = (
            np.sum(self.household_sector.theta) / self.equilibrium_budget_checker
        )
        self.equilibrium_utility = np.dot(
            self.household_sector.theta, np.log(self.equilibrium_consumptions)
        ) - self.household_sector.gamma * np.power(
            self.equilibrium_labour / self.household_sector.l_0,
            self.household_sector.phi + 1,
        ) / (
            self.household_sector.phi + 1
        )

    def save_eco(self, path_to_save) -> None:
        """
        Saves the economy as multi-indexed data-frame in hdf format along with networks in
        npy format.
        :param name: name of file,
        """
        first_index = np.concatenate(
            (np.repeat("Firms", 11), np.repeat("Household", 11))
        )
        second_index = np.concatenate(
            (
                [
                    "q",
                    "b",
                    "z",
                    "sigma",
                    "alpha",
                    "alpha_p",
                    "beta",
                    "beta_p",
                    "w",
                    "p_eq",
                    "g_eq",
                ],
                ["l", "theta", "gamma", "phi"],
            )
        )
        multi_index = [first_index, second_index]
        values = np.vstack(
            (
                self.ces_parameter * np.ones(self.firms_number),
                self.returns_to_scale * np.ones(self.firms_number),
                self.firms_sector.productivity_factors,
                self.firms_sector.depreciation_stock,
                self.firms_sector.price_surplus_coupling * np.ones(self.firms_number),
                self.firms_sector.price_profit_coupling * np.ones(self.firms_number),
                self.firms_sector.production_profit_coupling
                * np.ones(self.firms_number),
                self.firms_sector.production_surplus_coupling
                * np.ones(self.firms_number),
                self.firms_sector.w * np.ones(self.firms_number),
                self.equilibrium_prices,
                self.equilibrium_production_levels,
                self.household_sector.l_0 * np.ones(self.firms_number),
                self.household_sector.theta,
                self.household_sector.gamma * np.ones(self.firms_number),
                self.household_sector.phi * np.ones(self.firms_number),
            )
        )

        df_eco = pd.DataFrame(
            values, index=multi_index, columns=[np.arange(1, self.firms_number + 1)]
        )
        df_eco.to_hdf(path_to_save + "/eco.h5", key="df", mode="w")
        np.save(path_to_save + "/network.npy", self.augmented_adjacency_matrix)
        if self.ces_parameter != 0:
            np.save(
                path_to_save + "/sub_network.npy", self.augmented_substitution_matrix
            )

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
            np.divide(w, np.power(z_zeta, q) * np.power(u, q)), exponent
        )
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
            raise ValueError("x must be of even length")

        # pylint: disable=unbalanced-tuple-unpacking
        p, g = np.split(x, 2)
        z, v, m_cal, exponent, kappa = par
        v1 = np.multiply(z, np.multiply(p, 1 - np.power(g, exponent)))
        m1 = np.dot(m_cal, p)
        m2 = g * m1 - p * np.dot(m_cal.T, g)
        return np.concatenate((m1 - v1 - v, m2 - g * v + kappa))
