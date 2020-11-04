import warnings

from numba import jit

warnings.simplefilter("ignore")

import numpy as np
from tqdm import tqdm_notebook


class Dynamics(object):

    def __init__(self, e, t_max, step_size=None, boost=None, nu=None, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        if step_size:
            self.step_s = step_size
        else:
            eps = self.eco.get_eps_cal()
            if eps >= 5:
                self.step_s = 1
            elif 5 > eps >= 0.1:
                self.step_s = 1
            else:
                self.step_s = np.power(10, np.floor(np.log10(eps)))
                
        self.prices = np.zeros(((t_max + 1)/self.step_s, self.n))
        self.wages = np.zeros((t_max + 1)/self.step_s)
        self.prices_net = np.zeros(((t_max + 1)/self.step_s, self.n))
        self.prods = np.zeros(((t_max + 1)/self.step_s, self.n))
        self.prod_exp = np.zeros(((t_max + 1)/self.step_s, self.n))
        self.targets = np.zeros(((t_max + 1)/self.step_s, self.n))
        self.stocks = np.zeros(((t_max + 1)/self.step_s, self.n, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_exchange = np.zeros(((t_max + 1)/self.step_s, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros(((t_max + 1)/self.step_s, self.n + 1, self.n + 1))
        self.Q_opt = np.zeros(((t_max + 1)/self.step_s, self.n, self.n+1))
        self.Q_prod = np.zeros(((t_max + 1)/self.step_s, self.n, self.n + 1))
        self.Q_used = np.zeros(((t_max + 1)/self.step_s, self.n, self.n + 1))
        self.mu = np.zeros((t_max + 1)/self.step_s)
        self.budget = np.zeros((t_max + 1)/self.step_s)
        self.budget_res = np.zeros((t_max + 1)/self.step_s)
        self.labour = np.zeros((t_max + 1)/self.step_s)

        self.store = store
        self.boost = boost
        self.nu = nu

    def clear_all(self):
        self.prices = np.zeros(((self.t_max + 1)/self.step_s, self.n))
        self.wages = np.zeros((self.t_max + 1)/self.step_s)
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros(((self.t_max + 1)/self.step_s, self.n))
        self.prod_exp = np.zeros(((self.t_max + 1)/self.step_s, self.n))
        self.targets = np.zeros(((self.t_max + 1)/self.step_s, self.n))
        self.stocks = np.zeros(((self.t_max + 1)/self.step_s, self.n, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_exchange = np.zeros(((self.t_max + 1)/self.step_s, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros(((self.t_max + 1)/self.step_s, self.n + 1, self.n + 1))
        self.Q_opt = np.zeros(((self.t_max + 1)/self.step_s, self.n, self.n + 1))
        self.Q_prod = np.zeros(((self.t_max + 1)/self.step_s, self.n, self.n + 1))
        self.Q_used = np.zeros(((self.t_max + 1)/self.step_s, self.n, self.n + 1))
        self.mu = np.zeros((self.t_max + 1)/self.step_s)
        self.budget = np.zeros((self.t_max + 1)/self.step_s)
        self.budget_res = np.zeros((self.t_max + 1)/self.step_s)
        self.labour = np.zeros((self.t_max + 1)/self.step_s)

    def time_t_minus(self, t):
        """
        Performs all the actions of step t-. First the household is updated
        (budget, consumption and labour offer). Then, firms forecast profits and balance for the current
        time and compute a production target accordingly. They then compute and post their needs (both in
        goods and labor) as to reach this target.
        :param t: current time step
        :return: side-effect
        """

        # Firms
        self.supply = np.concatenate(([self.labour[t]], self.eco.firms.z * self.prods[t] + np.diagonal(self.stocks[t])))

        self.targets[t+1] = self.eco.firms.compute_targets(self.prices[t],
                                                      self.Q_demand[t-1],
                                                      self.supply,
                                                      self.prods[t]+self.boost
                                                           )

        self.Q_opt[t] = self.eco.firms.compute_optimal_quantities_firms(self.targets[t+1],
                                                                    self.prices[t],
                                                                    self.prices_net,
                                                                    self.stocks[t],
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    self.eco.j_a,
                                                                    self.eco.zeros_j_a,
                                                                    self.n
                                                                    )

        self.Q_demand[t, 1:, 0] = self.Q_opt[t, :, 0]
        self.Q_demand[t, 1:, 1:] = np.clip(self.Q_opt[t, :, 1:] - (self.stocks[t]-np.diagonal(self.stocks[t])*np.eye(self.n)),
                                         0,
                                         None)
        #self.Q_demand[t, 1:, 1:] += 0.00001 * (self.Q_demand[t,1:,1:]!=0)


    def time_t(self, t):
        """
        Performs all the actions of step t. First, an exchange period takes place where both firms and
        households may buy goods (including labor) in accordance to the supply constraint. Then, firms can
        compute their real profit and balance and update their prices (including wage) accordingly for the
        next time-step.
        :param t: current time-step
        :return: side-effect
        """

        self.Q_exchange[t, 1:, 0] = self.Q_demand[t, 1:, 0] * np.minimum(1, self.labour[t] / np.sum(self.Q_demand[t, 1:, 0]))

        self.budget[t] = self.budget_res[t] + np.sum(self.Q_exchange[t, 1:, 0])

        self.Q_demand[t, 0, 1:] = self.Q_demand[t, 0, 1:] * np.minimum(1, self.budget[t] / (
            self.budget_res[t]+self.labour[t]))

        self.demand = np.sum(self.Q_demand[t], axis=0)
        # Supply constraint
        self.s_vs_d = np.clip(self.supply / self.demand, None, 1)  # =1 if supply >= demand

        # Real trades according to the supply constraint

        self.Q_exchange[t, :, 1:] = np.matmul(self.Q_demand[t, :, 1:], np.diag(self.s_vs_d[1:]))


        self.Q_prod[t, :, 0] = self.Q_exchange[t, 1:, 0]
        self.Q_prod[t, :, 1:] = self.Q_exchange[t, 1:, 1:] + np.minimum(self.stocks[t] - np.diag(self.stocks[t])*np.eye(self.n), self.Q_opt[t, :, 1:])

        self.Q_used[t] = np.matmul(np.diag(np.nanmin(np.divide(self.Q_prod[t], self.eco.j_a), axis=1)),
                                   self.eco.j_a)


        self.budget_res[t+1] = self.budget[t] - np.dot(self.Q_exchange[t, 0, 1:], self.prices[t])
        self.tradereal = np.sum(self.Q_exchange[t], axis=0)

        # Prices and wage update
        self.profits, self.balance, self.cashflow, self.tradeflow = \
            self.eco.firms.compute_profits_balance(self.prices[t],
                                                   self.Q_exchange[t],
                                                   self.supply,
                                                   self.demand
                                                   )

        self.wages[t+1] = self.eco.firms.update_wages(self.balance[0], self.tradeflow[0])
        # self.utility[t] = self.eco.house.utility(self.Q_real[t, 0, 1:], self.Q_real[t, 1:, 0])

    def time_t_plus(self, t):
        """
        Performs all the actions of step t+. Production for the next time-step starts and inventories
        are compiled.
        :param t: current time-step
        :return: side-effect
        """

        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.profits,
                                                          self.balance,
                                                          self.cashflow,
                                                          self.tradeflow
                                                          ) / self.wages[t+1]

        self.budget[t] = self.budget[t] / self.wages[t+1]
        self.budget_res[t+1] = np.clip(self.budget_res[t+1], 0, None) / self.wages[t+1]
        # Clipping to avoid negative almost zero values
        self.prices_net = self.eco.compute_p_net(self.prices[t + 1])

        self.prods[t + 1] = self.eco.production_function(self.Q_prod[t])

        self.prod_exp[t + 1] = (1-self.nu) * self.prod_exp[t] + self.nu * self.prods[t+1]

        stocks_no_diagonal = self.stocks[t] - np.diag(self.stocks[t])*np.eye(self.n)

        self.stocks[t + 1] = (self.eco.q == 0) * ((np.heaviside(stocks_no_diagonal - self.Q_opt[t, :, 1:], 1)) * (stocks_no_diagonal - self.Q_used[t, :, 1:])
                                                    + (1 - np.heaviside(stocks_no_diagonal - self.Q_opt[t, :, 1:], 1)) * (self.Q_prod[t, :, 1:]- self.Q_used[t, :, 1:]))

        np.fill_diagonal(self.stocks[t + 1], self.supply[1:]-np.sum(self.Q_exchange[t, :, 1:], axis=0))

        self.stocks[t + 1] = np.matmul(self.stocks[t + 1], np.diag(1 - self.eco.firms.sigma))

        self.mu[t], self.Q_demand[t + 1, 0, 1:], self.labour[t+1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res[t + 1],
                                                             self.prices[t + 1],
                                                             self.balance[0],
                                                             self.tradeflow[0],
                                                             self.n
                                                             )

        # self.labour[t+1] = self.eco.house.l
        # self.Q_demand[t + 1, 0, 1:] = self.eco.house.kappa / self.eco.p_eq
        # self.Q_demand[t+1, 1:, 0] = self.eco.j_a[:, 0] * np.power(self.eco.g_eq, 1./self.eco.b)

    def discrete_dynamics(self, p0, w0, g0, t1, s0, B0):
        self.clear_all()
        # Initial conditions at t=0
        # Household
        self.wages[1] = w0
        self.budget_res[1] = B0 / w0

        self.prods[1] = g0
        self.prod_exp[1] = (1-self.nu )* g0 + self.nu * self.prod_exp[0]
        self.stocks[1] = s0
        self.prices[1] = p0 / w0
        self.prices_net = self.eco.compute_p_net(self.prices[1])
        self.mu[0], self.Q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res[1],
                                                             self.prices[1],
                                                             0,
                                                             1,
                                                             self.n
                                                             )

        # self.labour[1] = self.eco.house.l
        # self.Q_demand[1, 0, 1:] = self.eco.house.kappa / self.eco.p_eq
        # self.Q_demand[1, 1:, 0] = self.eco.j_a[:, 0] * np.power(self.eco.g_eq, 1. / self.eco.b)

        self.supply = np.concatenate([[self.labour[1]], self.eco.firms.z * g0 + np.diagonal(s0)])

        # Firms

        self.targets[2] = t1
        self.Q_opt[1] = self.eco.firms.compute_optimal_quantities_firms(self.targets[2],
                                                                    self.prices[1],
                                                                    self.prices_net,
                                                                    self.stocks[1],
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    self.eco.j_a,
                                                                    self.eco.zeros_j_a,
                                                                    self.n
                                                                    )

        self.Q_demand[1, 1:, 0] = self.Q_opt[1, :, 0]
        self.Q_demand[1, 1:, 1:] = np.clip(
            self.Q_opt[1, :, 1:] - (self.stocks[1] - np.diagonal(self.stocks[1]) * np.eye(self.n)),
            0,
            None)

        self.time_t(1)
        self.time_t_plus(1)
        t = 2 * self.step_s
        pbar = tqdm_notebook(total=(self.t_max + 1)/self.step_s)
        pbar.update(1)
        while t < (self.t_max + 1)/self.step_s:
            # print(t)
            self.time_t_minus(t)
            self.time_t(t)
            self.time_t_plus(t)
            t += self.step_s
            pbar.update(self.step_s)

        # self.prods = g0
        # self.targets = t1
        # self.budget_res = B0 / w0

    @staticmethod
    @jit
    def compute_prods(e, Q_real, tmax, n, g0):
        prods = np.zeros((tmax + 1, n))
        prods[1] = g0
        for t in range(1, tmax - 1):
            prods[t + 1, :], minQ = e.production_function(Q_real[t, 1:, :])
        return prods

    @staticmethod
    @jit
    def compute_profits_balance_cashflow_tradeflow(e, Q_real, Q_demand, prices, prods, stocks, labour, tmax, n):
        supply_goods = e.firms.z * prods + stocks
        demand = np.sum(Q_demand, axis=1)
        profits, balance, cashflow, tradeflow = np.zeros((tmax + 1, n)), np.zeros((tmax + 1, n + 1)), \
                                                np.zeros((tmax + 1, n)), np.zeros((tmax + 1, n + 1))
        for t in range(1, tmax):
            supply_t = np.concatenate(([labour[t]], supply_goods[t]))
            profits[t], balance[t], cashflow[t], tradeflow[t] = e.firms.compute_profits_balance(prices[t],
                                                                                                Q_real[t],
                                                                                                supply_t,
                                                                                                demand[t]
                                                                                                )
        return profits, balance, cashflow, tradeflow

    @staticmethod
    @jit
    def compute_utility(e, Q_real, tmax):
        utility = np.zeros(tmax + 1)
        for t in range(1, tmax):
            utility[t] = e.house.utility(Q_real[t, 0, 1:], Q_real[t, 1:, 0])
        return utility

    @staticmethod
    @jit
    def compute_targets(e, Q_demand, prices, tmax, n):
        targets = np.zeros((tmax + 1, n))
