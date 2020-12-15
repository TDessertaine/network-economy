import numpy as np
from scipy.optimize import fsolve


class Household(object):

    @staticmethod
    def fixed_point_mu(x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x

    def __init__(self, labour, theta, gamma, phi, w_p):
        """
        Set the fundamental parameters of the household
        :param labour: quantity of labour for phi --> \infty
        :param theta: vector of preferences for the goods
        :param gamma: aversion to labour parameter
        :param phi: concavity parameter
        """
        self.l = labour
        self.theta = theta
        self.thetabar = np.sum(self.theta)
        self.theta = theta / self.thetabar
        self.gamma = gamma
        self.phi = phi
        self.w_p = w_p
        self.v_phi = np.power(gamma, 1. / phi) / np.power(labour, 1 + 1. / phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def utility(self, consumption, working_hours):
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(np.sum(working_hours) / self.l,
                                                                                1. + self.phi) / (
                       1. + self.phi)

    def compute_demand_cons_labour_supply(self, budget, prices, demand, balance, tradeflow, n, lda):
        theta = self.theta * np.exp(-2*self.w_p * balance / tradeflow)
        b_new = budget + (1 - lda) * demand

        if self.phi == 1:
            mu = .5*(np.sqrt(np.power(b_new * self.v_phi, 2)
                               + 4 * self.v_phi * np.sum(theta) * lda**2)
                       - b_new * self.v_phi) / lda**2
        elif self.phi == np.inf:
            mu = np.sum(theta) / (self.l + budget)
        else:
            raise Exception('Not coded yet')
            # x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            # mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))

        return mu, theta / (mu * prices), np.power(mu * lda, 1. / self.phi) / self.v_phi


    @staticmethod
    def budget_constraint(budget, prices, offered_cons):
        b_vs_c = np.clip(budget / np.dot(offered_cons, prices), None, 1)
        cons_real = offered_cons * b_vs_c
        budget_res = budget - np.dot(cons_real, prices)
        return b_vs_c, cons_real, budget_res
