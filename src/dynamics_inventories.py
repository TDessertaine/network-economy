import warnings


# warnings.simplefilter("ignore")

import numpy as np
import pdb

class Dynamics(object):

    def __init__(self, e, t_max, nu, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        self.prices = np.zeros(t_max + 1)
        self.wages = np.zeros(t_max + 1)
        self.prices_net = np.zeros(t_max + 1)
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros(t_max + 1)
        self.prod_exp = np.zeros((self.t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_exchange = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.Q_opt = np.zeros((t_max + 1, self.n, self.n+1))
        self.Q_prod = np.zeros((t_max + 1, self.n, self.n + 1))
        self.Q_used = np.zeros((t_max + 1, self.n, self.n + 1))
        self.mu = np.zeros(t_max + 1)
        self.budget = np.zeros(t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(t_max + 1)
        self.utility = np.zeros(t_max + 1)

        self.store = store
        self.nu = nu

    def clear_all(self):
        self.prices = np.zeros(self.t_max + 1)
        self.wages = np.zeros(self.t_max + 1)
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros(self.t_max + 1)
        self.prod_exp = np.zeros((self.t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_exchange = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.Q_opt = np.zeros((self.t_max + 1, self.n, self.n+1))
        self.Q_prod = np.zeros((self.t_max + 1, self.n, self.n + 1))
        self.Q_used = np.zeros((self.t_max + 1, self.n, self.n + 1))
        self.mu = np.zeros(self.t_max + 1)
        self.budget = np.zeros(self.t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(self.t_max + 1)
        self.utility = np.zeros(self.t_max + 1)

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
        self.supply = np.array([self.labour[t], self.eco.firms.z * self.prods + self.stocks[t]])

        self.targets = self.eco.firms.compute_targets(self.prices[t],
                                                      self.Q_demand[t - 1],
                                                      self.supply,
                                                      self.prods
                                                      )
        self.Q_opt[t] = self.eco.firms.compute_optimal_quantities_firms(self.targets,
                                                                    self.prices[t],
                                                                    self.prices_net,
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    self.eco.j_a,
                                                                    self.eco.zeros_j_a,
                                                                    self.n
                                                                    )
        
        self.Q_demand[t, 1, 1] = np.clip(self.Q_opt[t, 0, 1] - self.stocks[t], 0, None)
        self.Q_demand[t, 1, 0] = self.Q_opt[t, 0, 0]

    def time_t(self, t):
        """
        Performs all the actions of step t. First, an exchange period takes place where both firms and
        households may buy goods (including labor) in accordance to the supply constraint. Then, firms can
        compute their real profit and balance and update their prices (including wage) accordingly for the
        next time-step.
        :param t: current time-step
        :return: side-effect
        """
        # Supply constraint coefficient
        self.demand = np.sum(self.Q_demand[t], axis=0)
        self.s_vs_d = np.clip(self.supply / self.demand, None, 1)  # =1 if supply >= constraint

        # Real work according to the labour supply constraint and associated budget
        self.Q_exchange[t, 1, 0] = self.Q_demand[t, 1, 0] * self.s_vs_d[0]
        
        self.budget[t] = self.budget_res + self.Q_exchange[t, 1, 0]
        
        # Real trades according to the supply constraint and associated "production associated" matrices
        self.Q_exchange[t, :, 1] = self.Q_demand[t, :, 1]*self.s_vs_d[1]
        
        self.Q_exchange[t, 0, 1] = self.Q_exchange[t, 0, 1] * np.minimum(1, self.budget[t] / (
            self.Q_exchange[t, 0, 1]*self.prices[t]))
        
        self.Q_prod[t, 0, 0] = self.Q_exchange[t, 1, 0]
        self.Q_prod[t, 0, 1] = self.Q_exchange[t, 1, 1] + np.minimum(self.stocks[t], self.Q_opt[t, 0, 1])

        
        self.Q_used[t] = np.nanmin(np.divide(self.Q_prod[t], self.eco.j_a))*self.eco.j_a
        
        
        # Results of exchange
        self.tradereal = np.sum(self.Q_exchange[t], axis=1)
        self.budget_res = self.budget[t] - np.dot(self.Q_exchange[t, 0, 1], self.prices[t])

        # Prices and wage update
        self.profits, self.balance, self.cashflow, self.tradeflow = \
            self.eco.firms.compute_profits_balance(self.prices[t],
                                                   self.Q_exchange[t],
                                                   self.supply,
                                                   self.demand
                                                   )

        self.wages[t+1] = self.eco.firms.update_wages(self.balance[0], self.tradeflow[0])
        self.utility[t] = self.eco.house.utility(self.Q_exchange[t, 0, 1], self.Q_exchange[t, 1, 0])

    def time_t_plus(self, t):
        """
        Performs all the actions of step t+. Production for the next time-step starts and inventories
        are compiled.
        :param t: current time-step
        :return: side-effect
        """
         # Scaling of the price of the good according to the wage update
        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.profits,
                                                          self.balance,
                                                          self.cashflow,
                                                          self.tradeflow
                                                          ) / self.wages[t+1]
        
        
        # In turn, scaling of budget and savings
        self.budget[t] = self.budget[t] / self.wages[t+1]
        self.budget_res = np.clip(self.budget_res, 0, None) / self.wages[t+1]  
                # clipping to avoid negative almost zero values
                 
        # Computation of the network price 
        self.prices_net = self.eco.compute_p_net(self.prices[t + 1])
        
        # Production 
        self.prods = self.eco.production_function(self.Q_prod[t])
        
        # Stock update 
        self.prod_exp[t + 1] = (1-self.nu) * self.prod_exp[t] + self.nu * self.prods

        self.stocks[t + 1] = self.supply[1]-np.sum(self.Q_exchange[t, :, 1])

        self.stocks[t + 1] = self.stocks[t + 1]*np.exp(-self.eco.firms.sigma)
        
        
        

        self.mu[t], self.Q_demand[t + 1, 0, 1], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[t + 1],
                                                             self.demand[0], 
                                                             self.balance[0], 
                                                             self.tradeflow[0], 
                                                             self.n, 
                                                             self.nu
                                                             )

    def discrete_dynamics(self, p0, w0, g0, t1, s0, B0):
        self.clear_all()
        # Initial conditions at t=0
        # Household
        self.wages[1] = w0
        self.budget_res = B0 / w0
        # Firm
        self.prods = g0
        self.stocks[1] = s0
        self.prices[1] = p0 / w0
        self.prices_net = self.eco.compute_p_net(self.prices[1])
        self.mu[0], self.Q_demand[1, 0, 1], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[1], 
                                                             self.demand[0], 
                                                             0, 
                                                             1, 
                                                             self.n, 
                                                             self.nu
                                                             )
        self.supply = np.array([self.labour[1], self.eco.firms.z * g0 + s0])

        # Firms
        self.targets = t1
        self.Q_opt[1] = self.eco.firms.compute_optimal_quantities_firms(self.targets,
                                                            self.prices[1],
                                                            self.prices_net,
                                                            self.eco.q,
                                                            self.eco.b,
                                                            self.eco.lamb_a,
                                                            self.eco.j_a,
                                                            self.eco.zeros_j_a,
                                                            self.n
                                                            )
        
        self.Q_demand[1, 1, 0] = self.Q_opt[1, 0, 1]
        self.Q_demand[1, 1, 1] = np.clip(self.Q_opt[1, 0, 1] - self.stocks[1], 0, None)                                                            
     
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
