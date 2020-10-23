import warnings


# warnings.simplefilter("ignore")

import numpy as np
import pdb

class Dynamics(object):

    def __init__(self, e, rho, t_max, store=None):
        self.eco = e
        self.rho = rho
        self.t_max = t_max
        self.n = self.eco.n
        self.prices = np.zeros((t_max + 1, self.n))
        self.wages = np.zeros(t_max + 1)
        self.prices_net = np.zeros((t_max + 1, self.n))
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_real = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(t_max + 1)
        self.budget = np.zeros(t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(t_max + 1)

        self.store = store

    def clear_all(self):
        self.prices = np.zeros((self.t_max + 1, self.n))
        self.wages = np.zeros(self.t_max + 1)
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((self.t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_real = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(self.t_max + 1)
        self.budget = np.zeros(self.t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(self.t_max + 1)

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
        self.supply = np.concatenate(([self.labour[t]], self.eco.firms.z * self.prods + self.stocks[t]))

        self.targets = self.eco.firms.compute_targets(self.prices[t],
                                                      self.Q_demand[t - 1],
                                                      self.supply,
                                                      self.prods
                                                      )
        self.Q_demand[t, :, 1:] = self.eco.firms.compute_demands_firms(**{'targets':self.targets,
                                                                    'prices':self.prices[t],
                                                                    'prices_net':self.prices_net,
                                                                    'q':self.eco.q,
                                                                    'b':self.eco.b,
                                                                    'lamb_a':self.eco.lamb_a,
                                                                    'n':self.n}
                                                                    ).T

    def time_t(self, t):
        """
        Performs all the actions of step t. First, an exchange period takes place where both firms and
        households may buy goods (including labor) in accordance to the supply constraint. Then, firms can
        compute their real profit and balance and update their prices (including wage) accordingly for the
        next time-step.
        :param t: current time-step
        :return: side-effect
        """

        self.demand = np.sum(self.Q_demand[t], axis=1)
        # Supply constraint
        self.s_vs_d = np.clip(self.supply / self.demand, None, 1)  # =1 if supply >= constraint

        # Real work according to the labour supply constraint and associated budget
        self.Q_real[t, 0, 1] = self.Q_demand[t, 0, 1] * self.s_vs_d[0]
        self.budget[t] = self.budget_res + self.Q_real[t, 0, 1]

        # Budget constraint
        offered_cons = self.Q_demand[t, 0, 1] * self.s_vs_d[1]
        self.b_vs_c, self.Q_real[t, 1, 0], self.budget_res = self.eco.house.budget_constraint(self.budget[t],
                                                                                               self.prices[t],
                                                                                               offered_cons
                                                                                               )

        # Real trades according to the supply constraint
        diag = np.clip((np.sum(self.supply)-self.rho*self.Q_demand[t, 1, 0])/(self.demand[1]-self.rho*self.Q_real[t, 1, 0]), None, 1)

        self.Q_real[t, 1, 1] = np.multiply(self.Q_demand[t, 1, 1], diag)
        
        # print(self.Q_real[t])
        self.tradereal = np.sum(self.Q_real[t], axis=0)

        # Prices and wage update
        self.profits, self.balance, self.cashflow, self.tradeflow = \
            self.eco.firms.compute_profits_balance(self.prices[t],
                                                   self.Q_real[t],
                                                   self.supply,
                                                   self.demand
                                                   )

        self.wages[t] = self.eco.firms.update_wages(self.balance[0], self.tradeflow[0])
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
                                                          ) / self.wages[t]

        self.budget[t] = self.budget[t] / self.wages[t]
        self.budget_res = np.clip(self.budget_res, 0, None) / self.wages[
            t]  # Clipping to avoid negative almost zero values
        self.prices_net = self.eco.compute_p_net(self.prices[t + 1])

        self.prods = self.eco.production_function(self.Q_real[t, :, 1])
        self.stocks[t + 1] = self.eco.firms.update_stocks(self.supply[1:],
                                                          self.tradereal[1:]
                                                          )

        self.mu[t], self.Q_demand[t + 1, 1, 0], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[t + 1],
                                                             )

    def discrete_dynamics(self, **kwargs):
        p0 = kwargs['p0']
        w0=  kwargs['w0']
        g0 = kwargs['g0']
        t1 = kwargs['t1']
        s0 = kwargs['s0']
        B0 = kwargs['B0']
        #n = kwargs['n']
        #print(kwargs)
        self.clear_all()
        # Initial conditions at t=0
        # Household
        self.wages[0] = w0
        self.budget_res = B0 / w0
        # Firm
        self.prods = g0
        self.stocks[1] = s0
        self.prices[1] = p0 / w0
        self.prices_net = self.eco.compute_p_net(self.prices[1])
        self.mu[0], self.Q_demand[1, 1, 0], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[1]
                                                             )
        self.supply = np.concatenate([[self.labour[1]], self.eco.firms.z * g0 + s0])

        # Firms
        fix={}
        self.targets = t1
        fix['targets'] = np.copy(self.targets)
        fix['prices'] = np.copy(self.prices[1])
        fix['prices_net'] = np.copy(self.prices_net)
        fix['q'] = self.eco.q
        fix['b'] = self.eco.b
        fix['lamb_a'] = np.copy(self.eco.lamb_a)
        fix['n'] = self.n
        #print(fix)
        self.Q_demand[1, :, 1] = self.eco.firms.compute_demands_firms(**fix)
                                                                    
     
        self.time_t(1)
   
        self.time_t_plus(1)
  
        t = 2
        while t < self.t_max:
            # print(t)
            self.time_t_minus(t)
            self.time_t(t)
            self.time_t_plus(t)
            t += 1

        self.prods = g0
        self.targets = t1
        self.budget_res = B0 / w0


    @staticmethod
    def compute_utility(e, Q_real, tmax):
        utility = np.zeros(tmax + 1)
        for t in range(1, tmax):
            utility[t] = e.house.utility(Q_real[t, 0, 1:], Q_real[t, 1:, 0])
        return utility

    @staticmethod
    def compute_targets(e, Q_demand, prices, tmax, n):
        targets = np.zeros((tmax + 1, n))
