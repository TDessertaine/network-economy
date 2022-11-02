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
The ``firms`` module
======================

This module declares the Firms class which model n firms.
The attributes of this class are all the fixed parameters defining firms.
The methods of this class encode the way firms update the varying quantities such as prices, productions etc...
"""

import numpy as np
import errors_strings


class Firms:
    def __init__(
        self,
        firms_number: int = None,
        productivity_factors: np.array = None,
        depreciation_stock: np.array = None,
        price_surplus_coupling: float = None,
        price_profit_coupling: float = None,
        production_profit_coupling: float = None,
        production_surplus_coupling: float = None,
        wage_labour_coupling: float = None,
    ) -> None:

        if firms_number is not None and firms_number <= 0:
            raise ValueError(errors_strings.FIRMS_NUMBER)

        if productivity_factors is not None and (productivity_factors < 0).any():
            raise ValueError(errors_strings.PRODUCTIVITY_FACTORS)

        if depreciation_stock is not None and (depreciation_stock < 0).any():
            raise ValueError(errors_strings.DEPRECIATION_FACTORS)

        if price_surplus_coupling is not None and price_surplus_coupling < 0:
            raise ValueError(errors_strings.ADJUSTMENT_SPEEDS)

        if price_profit_coupling is not None and price_profit_coupling < 0:
            raise ValueError(errors_strings.ADJUSTMENT_SPEEDS)

        if production_profit_coupling is not None and production_profit_coupling < 0:
            raise ValueError(errors_strings.ADJUSTMENT_SPEEDS)

        if production_surplus_coupling is not None and production_surplus_coupling < 0:
            raise ValueError(errors_strings.ADJUSTMENT_SPEEDS)

        if wage_labour_coupling is not None and wage_labour_coupling < 0:
            raise ValueError(errors_strings.ADJUSTMENT_SPEEDS)

        self.firms_number = firms_number if firms_number else 100
        self.productivity_factors = (
            productivity_factors if productivity_factors else np.ones(self.firms_number)
        )  # Productivity factors
        self.depreciation_stock = (
            depreciation_stock if depreciation_stock else np.ones(self.firms_number)
        )
        self.price_surplus_coupling = (
            price_surplus_coupling if price_surplus_coupling else 0.25
        )
        self.price_profit_coupling = (
            price_profit_coupling if price_profit_coupling else 0.25
        )
        self.production_profit_coupling = (
            production_profit_coupling if production_profit_coupling else 0.25
        )
        self.production_surplus_coupling = (
            production_surplus_coupling if production_surplus_coupling else 0.25
        )
        self.wage_labour_coupling = (
            wage_labour_coupling if wage_labour_coupling else 0.1
        )

    # Setters for class instances

    def update_productivity_factors(self, productivity_factors: np.array) -> None:
        if 
        self.productivity_factors = productivity_factors

    def update_sigma(self, sigma):
        self.depreciation_stock = sigma

    def update_alpha(self, alpha):
        self.price_surplus_coupling = alpha

    def update_alpha_p(self, alpha_p):
        self.price_profit_coupling = alpha_p

    def update_beta(self, beta):
        self.production_profit_coupling = beta

    def update_beta_p(self, beta_p):
        self.production_surplus_coupling = beta_p

    def update_w(self, omega):
        self.wage_labour_coupling = omega

    def update_prices(self, prices, profits, balance, cashflow, tradeflow, step_s):
        """
        Updates prices according to observed profits and balances.
        :param prices: current wage-rescaled prices,
        :param profits: current wages-rescaled profits,
        :param balance: current balance,
        :param cashflow: current wages-rescaled gain + losses,
        :param tradeflow: current supply + demand,
        :param step_s: size of time step,
        :return: Updated prices for the next period.
        """
        return prices * np.exp(
            -2
            * step_s
            * (
                self.price_profit_coupling * profits / cashflow
                + self.price_surplus_coupling * balance[1:] / tradeflow[1:]
            )
        )

    def update_wages(self, labour_balance, total_labour, step_s):
        """
        Updates wages according to the observed tensions in the labour market.
        :param labour_balance: labour supply - labour demand,
        :param total_labour: labour supply + labour demand,
        :param step_s: size of time-step,
        :return: Updated wage for the next period.
        """
        return np.exp(
            -2 * self.wage_labour_coupling * step_s * (labour_balance / total_labour)
        )

    def compute_targets(self, prices, q_forecast, supply, prods, step_s):
        """
        Computes the production target based on profit and balance forecasts.
        :param prices: current rescaled prices,
        :param q_forecast: forecast exchanged quantities,
        :param supply: current supply,
        :param prods: current production levels,
        :param step_s: size of time-step,
        :return: Production targets for the next period.
        """
        est_profits, est_balance, est_cashflow, est_tradeflow = self.compute_forecasts(
            prices, q_forecast, supply
        )
        return prods * np.exp(
            2
            * step_s
            * (
                self.production_profit_coupling * est_profits / est_cashflow
                - self.production_surplus_coupling * est_balance[1:] / est_tradeflow[1:]
            )
        )

    @staticmethod
    def compute_profits_balance(prices, q_exchange, supply, demand):
        """
        Compute the real profits and balances of firms.
        :param prices: current wage-rescaled prices,
        :param q_exchange: matrix of exchanged goods, labor and consumptions,
        :param supply: current supply,
        :param demand: current demand,
        :return: Realized wage-rescaled values of profits, balance, cash-flow, trade-flow.
        """
        gain = prices * np.sum(q_exchange[:, 1:], axis=0)
        losses = np.matmul(q_exchange[1:, :], np.concatenate((np.array([1]), prices)))

        return gain - losses, supply - demand, gain + losses, supply + demand

    @staticmethod
    def compute_optimal_quantities(targets, prices, e):
        """
        Computes minimizing-costs quantities given different production functions and production target.
        :param e: economy class,
        :param targets: production targets for the next period,
        :param prices: current wages-rescaled prices,
        :return: Matrix of optimal goods/labor quantities.
        """
        if e.q == 0:
            demanded_products_labor = np.matmul(
                np.diag(np.power(targets, 1.0 / e.b)), e.lamb_a
            )
            # print(demanded_products_labor)
        elif e.q == np.inf:
            prices_net_aux = np.array(
                [
                    np.prod(
                        np.power(
                            e.j_a[i, :]
                            * np.concatenate((np.array([1]), prices))
                            / e.a_a[i, :],
                            e.a_a[i, :],
                        )[e.zeros_j_a[i, :]]
                    )
                    for i in range(e.n)
                ]
            )
            demanded_products_labor = np.multiply(
                e.a_a,
                np.outer(
                    np.multiply(prices_net_aux, np.power(targets, 1.0 / e.b)),
                    np.concatenate((np.array([1]), 1.0 / prices)),
                ),
            )
        else:
            prices_net = np.matmul(
                e.lamb_a, np.power(np.concatenate(([1], prices)), e.zeta)
            )
            demanded_products_labor = np.multiply(
                e.lamb_a,
                np.outer(
                    np.multiply(
                        np.power(prices_net, e.q), np.power(targets, 1.0 / e.b)
                    ),
                    np.power(np.concatenate((np.array([1]), prices)), -e.q / (1 + e.q)),
                ),
            )
        return demanded_products_labor

    @staticmethod
    def compute_forecasts(prices, q_forecast, supply):
        """
        Computes the expected profits and balances assuming same demands as previous time.
        :param prices: current wage-rescaled prices,
        :param q_forecast: forecast exchanged quantities,
        :param supply: current supply,
        :return: Forecast of profits, balance, cash-flow and trade-flow.
        """
        exp_gain = prices * np.sum(q_forecast[:, 1:], axis=0)
        exp_losses = np.matmul(
            q_forecast[1:, :], np.concatenate((np.array([1]), prices))
        )
        exp_supply = supply
        exp_demand = np.sum(q_forecast, axis=0)
        return (
            exp_gain - exp_losses,
            exp_supply - exp_demand,
            exp_gain + exp_losses,
            exp_supply + exp_demand,
        )
