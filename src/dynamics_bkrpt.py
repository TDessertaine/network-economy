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
The ``dynamics`` module
======================

This module declares the Dynamics class which simulate the dynamics of the Network Economy ABM.
"""


import warnings
import numpy as np
import pandas as pd
from time import time

from numba import jit
from scipy.signal import periodogram
from scipy.special.cython_special import binom

warnings.simplefilter("ignore")


class Dynamics(object):
    def __init__(self, e, t_max, step_size=None, lda=None, nu=None, store=None,min_loss = None):
        """Create Dynamics' class to run the simulation.

        Args:
            e (economy.Economy): Economy from economy.py
            t_max (int): Max time of the simulation.
            step_size (float, optional): Time intervalle. Defaults to None.
            lda (float [0,1], optional): Initial convex coefficient for forecasting rule. Defaults to None.
            nu (float [0,1], optional): _description_. Defaults to None.
            store (bool, optional): _description_. Defaults to None.
            min_loss (int, optional): Profit loss rate under which a firm goes bankrupt.

        """
        self.eco = e  # Economy for which to run the simulations
        self.t_max = t_max  # End time of the simulation
        self.n = self.eco.n  # Number of firms
        self.n_= self.n
        self.min_loss = min_loss
        self.step_s = step_size if step_size else 1  # Size of on time step

        # Initialization of time-series
        self.prices = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.prices_non_res = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((t_max + 1) / self.step_s))
        self.prods = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.stocks = np.zeros((int((t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros((int((t_max + 1) / self.step_s),self.n))
        self.losses = np.zeros((int((t_max + 1) / self.step_s),self.n))
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((int((t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = 0
        self.savings = 0
        self.labour = np.zeros(int((t_max + 1) / self.step_s))
        self.contagion_matrix = np.ones((int((t_max + 1) / self.step_s),self.n))

        # Bankrupt idx
        self.srv_idx = np.arange(self.n)
        self.srv_idx_a = np.arange(self.n+1)
        self.bkrpt_idx = np.array([])
        self.n_bkrpt = np.zeros(int((t_max) / self.step_s))

        # Whether to store the dynamics in a h5 format
        self.store = store

        # Mask 
        self.init_mask()

        # Declare initial conditions instances
        self.p0 = None
        self.w0 = None
        self.g0 = None
        self.t1 = None
        self.s0 = None
        self.B0 = None

        self.run_with_current_ic = False
        self.nu = nu if nu else 1
        self.lda = lda if lda else 1
    
    def init_mask(self):
        # Mask
        # self.planning_mask_1 = np.array([[True if j in self.srv_idx_a and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)])                                            
        self.planning_mask_1 = np.ones((self.n,self.n+1),dtype = bool)
        
        #self.planning_mask_2 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx+1 else False for j in range(0,self.n+1)] for i in range(0,self.n+1)]) 
        self.planning_mask_2 = np.zeros((self.n+1,self.n+1),dtype = bool)
        self.planning_mask_2[1:,self.srv_idx+1] = True   
        
        #self.planning_mask_1 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx_a else False for j in range(0,self.n+1)] for i in range(0,self.n+1)]) 
        self.eau_mask_1 = np.concatenate([np.zeros(self.n+1,dtype = bool).reshape(-1,1),np.ones((self.n+1,self.n),dtype = bool)],axis = 1)


        # self.eau_mask_2 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        self.eau_mask_2 = np.zeros((self.n,self.n+1),dtype = bool)
        self.eau_mask_2[:,1:] = True

        # self.production_mask_1 = np.array([[True if j in self.srv_idx_a and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        self.production_mask_1 = np.ones((self.n,self.n+1),dtype = bool)

        # self.production_mask_2 = np.array([[True if j in self.srv_idx and i in self.srv_idx else False for j in range(0,self.n)] for i in range(0,self.n)]) 
        self.production_mask_2 = np.ones((self.n,self.n))

        #self.diag_mask = np.array([[True if (i,j) in list(zip(self.srv_idx,self.srv_idx)) else False for j in range(self.n)] for i in range(self.n)])
        self.diag_mask = np.identity(self.n,dtype = bool)


    def update_mask(self):
        # self.planning_mask_1 = np.array([[True if j in self.srv_idx_a and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)])                                            
        self.planning_mask_1[:,self.bkrpt_idx+1]   = False
        self.planning_mask_1[self.bkrpt_idx,:]     = False

        #self.planning_mask_2 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx+1 else False for j in range(0,self.n+1)] for i in range(0,self.n+1)]) 
        self.planning_mask_2[1:,self.bkrpt_idx+1] = False
        self.planning_mask_2[self.bkrpt_idx+1,1:] = False

        #self.eau_mask_1 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx_a else False for j in range(0,self.n+1)] for i in range(0,self.n+1)]) 
        self.eau_mask_1[:,self.bkrpt_idx+1] = False
        self.eau_mask_1[self.bkrpt_idx+1,1:] = False

        # self.eau_mask_2 = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        self.eau_mask_2[:,self.bkrpt_idx+1] = False
        self.eau_mask_2[self.bkrpt_idx,1:]  = False

        # self.production_mask_1 = np.array([[True if j in self.srv_idx_a and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        self.production_mask_1[:,self.bkrpt_idx+1] = False
        self.production_mask_1[self.bkrpt_idx,:]  = False

        # self.production_mask_2 = np.array([[True if j in self.srv_idx and i in self.srv_idx else False for j in range(0,self.n)] for i in range(0,self.n)]) 
        self.production_mask_2[:,self.bkrpt_idx] = False
        self.production_mask_2[self.bkrpt_idx,:]  = False

        #self.diag_mask = np.array([[True if (i,j) in list(zip(self.srv_idx,self.srv_idx)) else False for j in range(self.n)] for i in range(self.n)])
        self.diag_mask[self.bkrpt_idx,self.bkrpt_idx] = False

    def clear_all(self, t_max=None):
        """
        Clear every time-series in memory.
        :param t_max:
        :return: Emptied time-series instances.
        """
        self.n_ = self.n
        if t_max:
            self.t_max = t_max
        self.prices = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.prices_non_res = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((self.t_max + 1) / self.step_s))
        self.prods = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.stocks = np.zeros((int((self.t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros((int((self.t_max + 1) / self.step_s),self.n))
        self.losses = np.zeros((int((self.t_max + 1) / self.step_s),self.n))
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = 0
        self.savings = 0
        self.labour = np.zeros(int((self.t_max + 1) / self.step_s))
        self.contagion_matrix = np.ones((int((self.t_max + 1) / self.step_s),self.n))

        self.srv_idx = np.arange(self.n)
        self.srv_idx_a = np.arange(self.n+1)
        self.bkrpt_idx = np.array([])
        self.n_bkrpt = np.zeros(int((self.t_max) / self.step_s))


        self.init_mask()
    # Setters for simulation parameters


    def update_min_loss(self,min_loss):
        self.min_loss = min_loss

    def update_tmax(self, t_max):
        self.clear_all(t_max)
        self.store = self.store
        self.run_with_current_ic = False

    def update_step_size(self, step_size):
        self.step_s = step_size
        self.clear_all(self.t_max)
        self.run_with_current_ic = False

    # Setters for the economy

    def update_eco(self, e):
        self.eco = e

    # Setters for initial conditions

    def set_initial_conditions(self, p0, w0, g0, t1, s0, B0):
        self.p0 = p0
        self.w0 = w0
        self.g0 = g0
        self.t1 = t1
        self.s0 = s0
        self.B0 = B0
        self.run_with_current_ic = False

    def set_initial_price(self, p0):
        self.p0 = p0
        self.run_with_current_ic = False

    def set_initial_wage(self, w0):
        self.w0 = w0
        self.run_with_current_ic = False

    def set_initial_prod(self, g0):
        self.g0 = g0
        self.run_with_current_ic = False

    def set_initial_target(self, t1):
        self.t1 = t1
        self.run_with_current_ic = False

    def set_initial_stock(self, s0):
        self.s0 = s0
        self.run_with_current_ic = False

    def set_initial_budget(self, B0):
        self.B0 = B0
        self.run_with_current_ic = False

    # Dynamical methods

    def planning(self, t):
        """
        Performs all the actions the Planning step. Firms forecast gains, losses, supplies and demands (in the
        compute_targets function) and compute the subsequent targets. Then, they post their demands (goods and labor)
        for the current period.
        :param t: current time step
        :return: side-effect
        """

        # (1) - (2) Forecasts and production targets
        self.supply[self.srv_idx_a] = np.concatenate(
            ([self.labour[t]], self.eco.firms.z[self.srv_idx] * self.prods[t,self.srv_idx] + np.diagonal(self.stocks[t,self.srv_idx,:][:,self.srv_idx])))

        self.targets[t + 1,self.srv_idx] = self.eco.firms.compute_targets(self.prices[t,self.srv_idx],
                                                             self.lda * self.q_demand[t - 1][self.srv_idx_a,:][:,self.srv_idx_a] +
                                                             (1 - self.lda) * self.q_exchange[t - 1][self.srv_idx_a,:][:,self.srv_idx_a],
                                                             self.supply[self.srv_idx_a],
                                                             self.prods[t,self.srv_idx],
                                                             self.step_s
                                                             )
        np.place(self.q_opt,self.planning_mask_1,self.eco.firms.compute_optimal_quantities(self.targets[t + 1,self.srv_idx],
                                                               self.prices[t,self.srv_idx],
                                                               self.eco,
                                                               self.srv_idx,
                                                               self.srv_idx_a
                                                               ))

        # (3) Posting demands
        
        self.q_demand[t, self.srv_idx+1, 0] = self.q_opt[self.srv_idx, 0]
        np.place(self.q_demand[t],self.planning_mask_2,np.maximum(
            self.q_opt[self.srv_idx,:][:,self.srv_idx+1] - (self.stocks[t][self.srv_idx,:][:,self.srv_idx] - np.diagonal(self.stocks[t][self.srv_idx,:][:,self.srv_idx]) * np.eye(self.n_)),
            0))

    def exchanges_and_updates(self, t):
        """
        Performs all the actions the Exchanges & Trades step. Hiring occurs first, upon which the household gets paid.
        It allows it to compute its current budget and adapt its consumption target accordingly. Then, we move on to
        the trades where firms buy from one another and the household consumes (and computes its subsequent savings).
        Finally, everything is known for the firms to compute profits and balances and update prices and wage.
        :param t: current time-step
        :return: side-effect
        """

        # (1) Hiring and Wage payment
        self.q_exchange[t, self.srv_idx+1, 0] = self.q_demand[t, self.srv_idx+1, 0] * np.minimum(1, self.labour[t] / np.sum(
            self.q_demand[t, self.srv_idx+1, 0]))

        self.budget = self.savings + np.sum(self.q_exchange[t, self.srv_idx+1, 0])

        self.q_demand[t, 0,self.srv_idx+1] = self.q_demand[t, 0,self.srv_idx+1] * (self.nu + (1 - self.nu) *
                                                             np.minimum(1, self.budget /
                                                                        (self.savings + self.labour[t])))

        # (2) Trades
        self.demand[self.srv_idx_a] = np.round(np.sum(self.q_demand[t][self.srv_idx_a,:][:,self.srv_idx_a], axis=0),10)
        
        np.place(self.q_exchange[t],self.eau_mask_1,np.matmul(self.q_demand[t][self.srv_idx_a,:][:,self.srv_idx+1],
                                              np.diag(np.minimum(self.supply[self.srv_idx+1] / self.demand[self.srv_idx+1], 1))
                                              ))
    

        self.q_exchange[t, 0, self.srv_idx+1] = self.q_exchange[t, 0, self.srv_idx+1] * np.minimum(1, self.eco.house.f * self.budget / (
            np.dot(self.q_exchange[t, 0, self.srv_idx+1], self.prices[t,self.srv_idx])))
    
        self.savings = self.budget - np.dot(self.prices[t,self.srv_idx], self.q_exchange[t,0,self.srv_idx+1])
        
        self.q_prod[self.srv_idx, 0] = self.q_exchange[t][self.srv_idx+1,0]
        # mask_2D = np.array([[True if j in self.srv_idx+1 and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        np.place(self.q_prod,self.eau_mask_2,self.q_exchange[t][self.srv_idx+1,:][:,self.srv_idx+1] + np.minimum(
            self.stocks[t][self.srv_idx,:][:,self.srv_idx] - np.diag(self.stocks[t][self.srv_idx,:][:,self.srv_idx]) * np.eye(self.n_), self.q_opt[self.srv_idx,:][:,self.srv_idx+1]))

        self.tradereal = np.sum(self.q_exchange[t,self.srv_idx_a], axis=0)
        
        self.gains[t,self.srv_idx] = self.prices[t,self.srv_idx] * self.tradereal[self.srv_idx+1]
        self.losses[t,self.srv_idx] = np.round(np.matmul(self.q_exchange[t][self.srv_idx+1,:][:,self.srv_idx_a], np.concatenate(([1], self.prices[t,self.srv_idx]))),10)

    
        # (3) Prices and Wage updates
        self.wages[t + 1] = self.eco.firms.update_wages(self.supply[0] - self.demand[0],
                                                        self.supply[0] + self.demand[0],
                                                        self.step_s)

        self.prices[t + 1,self.srv_idx] = self.eco.firms.update_prices(self.prices[t,self.srv_idx],
                                                      self.gains[t,self.srv_idx] - self.losses[t,self.srv_idx],
                                                      self.supply[self.srv_idx_a] - self.demand[self.srv_idx_a],
                                                      self.gains[t,self.srv_idx] + self.losses[t,self.srv_idx],
                                                      self.supply[self.srv_idx_a] + self.demand[self.srv_idx_a],
                                                      self.step_s
                                                      )

    def production(self, t):
        """
        Performs all the actions of the Production Step. Production for the next time-step starts, inventories
        are compiled and monetary quantities are rescaled by the wage value. Then, the household then performs its
        optimization thus setting consumption targets and labour supply for the next period.
        :param t: current time-step
        :return: side-effect
        """

        # (1) Production starts
        self.prods[t + 1,self.srv_idx] = self.eco.production_function(self.q_prod[self.srv_idx,:][:,self.srv_idx_a],self.srv_idx,self.srv_idx_a)
        
        # mask_2D = np.array([[True if j in self.srv_idx_a and i in self.srv_idx else False for j in range(0,self.n+1)] for i in range(0,self.n)]) 
        np.place(self.q_used,self.production_mask_1,(self.eco.q == 0) * np.matmul(np.diag(np.nanmin(np.divide(self.q_prod[self.srv_idx,:][:,self.srv_idx_a], self.eco.j_a[self.srv_idx,:][:,self.srv_idx_a]), axis=1)),
                                                    self.eco.j_a[self.srv_idx,:][:,self.srv_idx_a]) + (self.eco.q != 0) * self.q_prod[self.srv_idx,:][:,self.srv_idx_a])
        # (2) Inventory update
        # mask_2D = np.array([[True if j in self.srv_idx and i in self.srv_idx else False for j in range(0,self.n)] for i in range(0,self.n)]) 

        np.place(self.stocks[t + 1],self.production_mask_2,(self.eco.q == 0) * (self.q_prod[self.srv_idx,:][:,self.srv_idx+1] - self.q_used[self.srv_idx,:][:,self.srv_idx+1]))
        np.place(self.stocks[t + 1],self.diag_mask,self.supply[self.srv_idx+1] - self.tradereal[self.srv_idx+1])
        np.place(self.stocks[t + 1],self.production_mask_2,np.matmul(self.stocks[t + 1][self.srv_idx,:][:,self.srv_idx], np.diag(np.exp(- self.eco.firms.sigma[self.srv_idx] * self.step_s))))

    #     # (3) Price rescaling
        self.prices[t+1,self.srv_idx]=self.prices[t+1,self.srv_idx]/ self.wages[t + 1]
        self.budget = self.budget / self.wages[t + 1]
        self.savings = (1 + self.eco.house.r) * np.maximum(self.savings, 0) / self.wages[t + 1]
    #     # Clipping to avoid negative almost zero values

    #     # The household performs its optimization to set its consumption target and its labour supply for the next
    #     # period
        self.q_demand[t + 1, 0, self.srv_idx+1], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.savings,
                                                            self.prices[t + 1,self.srv_idx],
                                                            self.supply[0],
                                                            self.demand[0],
                                                            self.step_s,
                                                            self.srv_idx
                                                            )

    def banrkuptcy(self,t):
        """Update bankruptcy

        Args:
            t (_type_): _description_
        """
         # Update temporary information.
        cum_profits = (self.gains-self.losses).cumsum(axis = 0)
        bkrpt_idx = np.array(np.where(cum_profits[t]<-self.min_loss))[0]
        new_idx   = set(bkrpt_idx)-set(np.array(np.where(cum_profits[t-1]<-self.min_loss))[0])
        if new_idx :
            self.bkrpt_idx = np.unique(np.concatenate([self.bkrpt_idx,bkrpt_idx])).astype(int) #Update index of bunkrapted firms with firms previously died (bkrpt + cascade) and firms that bkrpt at time t
            self.srv_idx   = np.array(list(set(self.srv_idx)-set(self.bkrpt_idx)))
            self.bkrpt_idx,self.srv_idx = self.eco.update_j(self.bkrpt_idx,self.srv_idx) # Update economy information (link, preferencies...)
            self.n_ = len(self.srv_idx)
            self.contagion_matrix[t,self.bkrpt_idx] = 0
            self.srv_idx_a = np.concatenate([[0],self.srv_idx+1])
            self.update_mask()
        self.n_bkrpt[t] = len(self.bkrpt_idx)
        return self.n_ == 0

    def discrete_dynamics(self):
        """
        Main function to run the dynamics of the Network Economy ABM.
        :return: Side-effect
        """
        self.clear_all()

        # Setting initial conditions
        self.wages[1] = self.w0
        self.savings = self.B0 / self.w0

        self.prods[1] = self.g0
        self.stocks[1] = self.s0
        self.prices[1] = self.p0 / self.w0
        self.prices_non_res[1] = self.p0
        self.q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.savings,
                                                             self.prices[1],
                                                             1,
                                                             1,
                                                             self.step_s
                                                             )

        # Planning period with provided initial target t1.
        self.supply = np.concatenate([[self.labour[1]], self.eco.firms.z * self.g0 + np.diagonal(self.s0)])
        self.targets[2] = self.t1
        self.q_opt = self.eco.firms.compute_optimal_quantities(self.targets[2],
                                                               self.prices[1],
                                                               self.eco
                                                               )

        self.q_demand[1, 1:, 0] = self.q_opt[:, 0]
        self.q_demand[1, 1:, 1:] = np.maximum(
            self.q_opt[:, 1:] - (self.stocks[1] - np.diagonal(self.stocks[1]) * np.eye(self.n)),
            0)

        # Carrying on with Exchanges & Trades and Production with every needed quantities known.
        self.exchanges_and_updates(1)
        self.production(1)
        test = self.banrkuptcy(1)
        # End of first time-step
        t = 2
        self.time_table = np.zeros((int((self.t_max) / self.step_s),4))
        while t < int((self.t_max + 1) / self.step_s - 1) and test == False:
            t1 = time()
            self.planning(t)
            t2 = time()
            self.exchanges_and_updates(t)
            t3 = time()
            self.production(t)
            t4 = time()
            test = self.banrkuptcy(t)
            t5 = time()
            self.time_table[t] = np.array([t2-t1,t3-t2,t4-t3,t5-t4])
            t += 1

        # The current information stocked in the dynamics class are in accordance with the provided initial conditions.
        self.run_with_current_ic = True

    # Classification methods

    def norm_prices_prods_stocks(self):
        """
        :return: A data-frame of prices, productions and diagonal stocks across time.
        """
        dfp = pd.DataFrame(self.prices[1:-1] - self.eco.p_eq, columns=['p' + str(i) for i in range(self.n)])
        dfg = pd.DataFrame(self.prods[1:-1] - self.eco.g_eq, columns=['g' + str(i) for i in range(self.n)])
        dfs = pd.DataFrame([np.diagonal(s) for s in self.stocks[1:-1]], columns=['s' + str(i) for i in range(self.n)])
        df = pd.concat([dfp, dfg, dfs], axis=1)
        df = df.apply(lambda x: np.linalg.norm(x), axis=1)
        return df

    @staticmethod
    def rolling_diff(data, step_back):
        """
        :param data: data-frame on which to perform the rolling diff.
        :param step_back: window on which to perform the rolling diff.
        :return: Rolling min-max diff of data on a given window.
        """
        t_diff = []
        for t in range(1, len(data) - step_back + 1):
            t_diff.append(np.amax(data[- t - step_back:- t]) - np.amin(data[- t - step_back:- t]))
        df_t_diff = pd.DataFrame(t_diff[::-1])
        return df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0], (df_t_diff.iloc[-1] < 10e-8)[0]

    @staticmethod
    def fisher_test(data):
        """
        Perform a Fisher test on data-frame to determine if iy oscillates.
        References: Ahdesmäki M, Lähdesmäki H, Pearson R, Huttunen H, Yli-Harja O.
                    Robust detection of periodic time series measured from biological systems.
                    BMC Bioinformatics. 2005;6:117. Published 2005 May 13. doi:10.1186/1471-2105-6-117
        :param data: data-frame on which to perform the Fisher test.
        :return: p-value computed for the Fisher test.
        """
        freq, dft = periodogram(data, fs=1)
        q = int((len(data) - 1) / 2)
        stat = max(dft) / np.sum(dft)
        b = int(1 / stat)
        binom_vec = np.vectorize(lambda j: binom(q, j))
        j_vec = np.arange(b + 1)
        p_value = 1 - np.sum(np.power(-1, j_vec) * binom_vec(j_vec) * np.power(1 - j_vec * stat, q - 1))
        return p_value

    def detect_periodicity(self, data):
        """
        Function used to determine whether data is periodic using a Fisher test at level 0.001. We exclude extreme
        oscillations along with constant functions.
        :param data: data for which to check periodicity.
        :return: True if periodic.
        """
        return self.fisher_test(data) < 0.001 and 10e8 > data.var() > 10e-8

    def detect_convergent(self, data):
        """
        Function to determine whether data is convergent using rolling min-max.
        :param data: data for which to check convergency.
        :return: 3-tuple: True if convergent, True is convergent and oscillating, True if convergent towards 0.
        """
        bools = self.rolling_diff(data, 10)
        return bools[0], bools[1], data.iloc[-1] < 10e-8

    @staticmethod
    def detect_divergent(data):
        """
        Function used to determine if data diverges using rolling min-max. We consider divergent data for which any of
        the rolling min-max value is nan.
        :param data: data for which to check divergency.
        :return: True if prices diverge, False otherwise
        """
        if np.isnan(data).any():
            return True
        else:
            t_diff = []
            for t in range(1, len(data) - 10 + 1):
                t_diff.append(np.amax(data[- t - 10:- t]) - np.amin(data[- t - 10:- t]))
            df_t_diff = pd.DataFrame(t_diff[::-1])
            return np.isnan(df_t_diff).any()[0] or (df_t_diff.iloc[-1] > 10e6)[0]

    def detect_crises(self, data):
        """
        ! Need to work on this function. !
        Function used to determine if data exhibits a crises-like patterns.
        :param data: data for which to check crises-like patterns.
        :return: True if data is both convergent and relaxes towards 0.
        """
        return self.fisher_test(data) < 0.001 and data.min() < 10e-4

    # Reconstruction methods

    @staticmethod
    @jit
    def compute_gains_losses_supplies_demand(e, q_demand, q_exchange, prices, prods, stocks, labour):
        """
        Reconstruction method to compute gains, losses, supplies and demands across time.
        :param e: economy class,
        :param q_demand: time-series of demand matrices,
        :param q_exchange: time-series of exchange matrices,
        :param prices: time-series of wage-rescaled prices,
        :param prods: time-series of production levels
        :param stocks: time-series of stocks,
        :param labour: time-series of labour supply.
        :return: Time-series of computed gains, losses, supplies and demands.
        """
        demands = np.sum(q_demand, axis=1)
        # gains = np.zeros((len(prices), e.n))
        # losses = np.zeros((len(prices), e.n))
        supplies = np.zeros((len(prices), e.n + 1))
        for i in range(len(prices)):
            # gains[i] = prices[i] * np.sum(q_exchange[i, :, 1:], axis=0)
            # losses[i] = np.matmul(q_exchange[i, 1:, :], np.concatenate(([1], prices[i])))
            supplies[i] = np.concatenate(([labour[i]], e.firms.z * prods[i] + np.diag(stocks[i])))
        return supplies, demands

    @staticmethod
    @jit
    def compute_utility_budget(e, q_exchange, prices, rescaling_factors, t_max, step_s, initial_savings):
        """
        Reconstruction method to compute utility and non-rescaled budget across time.
        :param e: economy class,
        :param q_exchange: time-series of exchange matrices,
        :param prices: time-series of prices,
        :param rescaling_factors: time-series of wages used to rescale prices,
        :param t_max: maximum number of time-steps,
        :param step_s: size of one time step,
        :param initial_savings: initial savings
        :return: Time-series for utility and non-rescaled budgets.
        """
        utility = np.zeros(int((t_max + 1) / step_s))
        budget = np.zeros(int((t_max + 1) / step_s))
        budget[0] = initial_savings
        for i in range(1, int((t_max + 1) / step_s) - 1):
            utility[i] = np.power(rescaling_factors[i + 1], e.house.omega_p / e.firms.omega) * \
                         np.dot(e.house.theta, q_exchange[i, 0, 1:]) - \
                         np.power(np.sum(q_exchange[i, 1:, 0]) / e.house.l_0, 1 + e.house.phi) * e.house.gamma / (
                                 1 + e.house.phi)
            budget[i] = budget[i - 1] - np.dot(prices[i - 1], q_exchange[i - 1, 0, 1:]) + np.sum(q_exchange[i, 1:, 0])
        return utility, budget
    
    def visual(self,t):
        return {"Time":t,
        "n":self.n,
        "prices": self.prices[t],
        "prices_non_res": self.prices_non_res[t],
        "prods": self.prods[t],
        "targets": self.targets[t],
        "stocks": self.stocks[t],
        "gains": self.gains[t],
        "losses": self.losses[t],
        "supply": self.supply[t],
        "demand": self.demand[t],
        "tradereal": self.tradereal,
        "q_exchange": self.q_exchange[t],
        "q_demand": self.q_demand[t],
        "q_opt": self.q_opt,
        "q_prod": self.q_prod,
        "q_used": self.q_used}
