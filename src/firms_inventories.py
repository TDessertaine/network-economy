import numpy as np


# +
#from exception import *

# +
#spec = [
#    ('z', numba.float32[:]),
#    ('sigma', numba.float32[:]),
#    ('alpha', numba.float32),
#    ('alpha_p', numba.float32),
#    ('beta', numba.float32),
#    ('beta_p', numba.float32),
#    ('w', numba.float32)
#]
# -

class Firms(object):
    def __init__(self, z, sigma, alpha, alpha_p, beta, beta_p, w):


        # Production function parameters
        self.z = z
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.beta = beta
        self.beta_p = beta_p
        self.w = w

    def update_prices(self, prices, profits, balance, cashflow, tradeflow):
        """
        Updates prices according to observed profits and balances
        :param prices: current wage-rescaled prices
        :param profits: current wages-rescaled profits
        :param balance: current balance
        :param cashflow: current wages-rescaled gain + losses
        :param tradeflow: current supply + demand
        :return:
        """
        return prices * np.exp( -2*self.alpha_p * (profits / cashflow) - 2*self.alpha * (balance[1] / tradeflow[1]))

    def update_wages(self, labour_balance, total_labour):
        """
        Updates wages according to the observed tensions in the labour market
        :param labour_balance: labour supply - labour demand
        :param total_labour: labour supply + labour demand
        :return: Updated wage
        """
        return np.exp( - 2*self.w * labour_balance / total_labour)

    def compute_targets(self, prices, Q_demand_prev, supply, prods):
        """
        Computes the production target based on profit and balance forecasts.
        :param prices: current rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption demands
        :param supply: current supply
        :param prods: current production levels
        :return: Production targets for the next period
        """
        est_profits, est_balance, est_cashflow, est_tradeflow = self.compute_forecasts(prices, Q_demand_prev, supply)
        return prods * np.exp(2*self.beta * (est_profits / est_cashflow) - 2*self.beta_p * (est_balance[1] / est_tradeflow[1]))

   
    
    def compute_optimal_quantities_firms(self, targets, prices, prices_net, q, b, lamb_a, j_a, zeros_j_a, n):
        """
        Computes
        :param targets: production targets for the next period
        :param prices_net: current wage-rescaled aggregated network prices
        :param prices: current wages-rescaled aggregated network prices
        :param q: CES interpolator
        :param b: Return to scale parameter
        :param lamb_a: Aggregated network-substitution matrix
        :return: (n, n+1) matrix of labour/goods demands
        """
        if q == 0:
            demanded_products_labor = np.power(targets, 1. / b)*lamb_a
        elif q == np.inf:
            prices_net_aux = np.array([
                np.prod(np.power(j_a[1, :] * np.concatenate((np.array([1]), [prices])), lamb_a[1, :])[zeros_j_a[1, :]])])
            demanded_products_labor = np.multiply(lamb_a,
                                                  np.outer(np.multiply(prices_net_aux,
                                                                       np.power(targets, 1. / b)),
                                                           np.concatenate((np.array([1]), [1. / prices]))
                                                           ))
        else:
            demanded_products_labor = np.multiply(lamb_a,
                                                  np.outer(np.multiply(np.power(prices_net, q),
                                                                       np.power(targets, 1. / b)),
                                                           np.power(np.concatenate(([1], [prices])),
                                                                    - q / (1 + q))))
        return demanded_products_labor   
    
    
    
    @staticmethod
    def compute_forecasts(prices, Q_demand_prev, supply):
        """
        Computes the expected profits and balances assuming same demands as previous time
        :param prices: current wage-rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption demands
        :param supply: current supply
        :return: Forecasts of gains - losses, supply - demand, gains + losses, supply + demand
        """

        exp_gain = np.multiply(prices, np.sum(Q_demand_prev[:, 1:]))
        exp_losses = np.matmul(Q_demand_prev[1:, :], np.concatenate(([1], [prices])))
        exp_supply = supply
        exp_demand = np.sum(Q_demand_prev[:, 1])
        return exp_gain - exp_losses, exp_supply - exp_demand, exp_gain + exp_losses, exp_supply + exp_demand


    @staticmethod
    def compute_profits_balance(prices, Q, supply, demand):
        """
        Compute the real profits and balances of firms
        :param prices: current wage-rescaled prices
        :param Q: (n+1, n+1) matrix of exchanged goods, labour and consumption
        :param supply: current supply
        :param demand: current demand
        :return: Real wage-rescaled values of gains - losses, supply - demand, gains + losses, supply + demand
        """
        gain = np.multiply(prices, np.sum(Q[:, 1]))
        losses = np.matmul(Q[1, :], np.concatenate(([1], [prices])))

        return gain - losses, supply - demand, gain + losses, supply + demand
