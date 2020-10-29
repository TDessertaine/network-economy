#!/usr/bin/env python
# coding: utf-8
# %%
"""
Functions needed to run networks simulations in an aggregate manner:
- stocks simulation variables
- runs simulation
- computes and plots equilibria
"""

# %%
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from dynamics import Dynamics as dyn
from economy import Economy as eco

# %%
### Stocks simulation variables changing only the parameters you want to work on
def variables_simulation(alpha, alpha_p, beta, beta_p, w, q, b, p_init, g_init, pert):
    """
    Function used for stocking every parameter of a given simulation when
    iterating on the model's timescales.
    :param alpha: float, model's alpha
    :param alpha_p: float, model's alpha'
    :param beta: float, model's beta
    :param beta_p: float, model's beta_p
    :param w: float, model's w
    :param q: float, model's q
    :param b: float, model's b
    :param pert: float, perturbation applied on the price at t = 0
    :return: sim_args, dictionary of dictionaries of the simulation's params.
    """
    sim_args = {}
    # Variables statiques Firms
    sim_args["firms_args"] = {
        "z":np.random.uniform(12, 12, 1),
        "sigma":np.random.uniform(0.2, 0.2, 1),
        "alpha":alpha,
        "alpha_p":alpha_p,
        "beta":beta,
        "beta_p":beta_p,
        "w":w
        }
    # Variables statiques Household
    sim_args["house_args"] = {
        'labour':10,
        'theta':np.ones(1),
        'gamma':1,
        "phi":1
        }
    # Variables statiques Economiy
    sim_args["econ_args"] = {
        'n':1,
        'd':1,
        'netstring':'regular',
        'directed':True,
        'j0':np.array([5]),
        'j1':np.array([2]),
        'a0':np.ones(1)*0.5,
        'q':q,
        'b':b
        }
    # Variables statiques Dynamics
    sim_args["dyn_args"] = {
        'p0':np.array([p_init+pert]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':g_init+pert,
        's0':np.random.uniform(0, 0, 1),
        't1':g_init+pert,
        'B0':random.randint(0, 0)
        }
    return sim_args

# %%
### Initializes and runs simulation
def simulation(**sim_args):
    """
    Function used for launching a simulation.
    :dic sim_args: dictionary of parameters for the simulation
    :return: Dynamics object sim
    """
# Création objet classe Economy
    economie = eco(**sim_args["econ_args"])
    # Initialisations demandées à Economy
    economie.init_house(**sim_args["house_args"])
    economie.init_firms(**sim_args["firms_args"])
    economie.set_quantities()
    # Création de l'objet dynamique
    sim = dyn(t_max=10000, e=economie, rho=0)
    # Dynamique
    sim.discrete_dynamics(**sim_args["dyn_args"])
    return sim

# %%
### Disturbs equilibrium
def disturbs_equilibrium(sim, pert_p, pert_g):
    """
    Function used for disturbing the economy away from its
    equilibrium price and production.
    :param sim: Dynamics object
    :return: tuple p_eq__dis_0, g_eq_dis_0, perturbed price and prod level
    """
    sim.eco.compute_eq()
    p_eq_dis_0=sim.eco.p_eq[0] + pert_p
    g_eq_dis_0=sim.eco.g_eq[0] + pert_g
    return p_eq_dis_0, g_eq_dis_0


# %%
### Computes and stocks equilibrium values
def compute_equilibrium(sim):
    """
    Function used for computing the economy's equilibrium price and production.
    :param sim: Dynamics object
    :return: tuple p_eq_0, g_eq_0, the economy's equilibrium values
    """
    sim.eco.compute_eq()
    print("P_EQ", sim.eco.p_eq)
    print("G_EQ", sim.eco.g_eq)
    if len(sim.eco.p_eq)==1:
        p_eq_0 = sim.eco.p_eq[0]
        g_eq_0 = sim.eco.g_eq[0]
        if sim.eco.b != 1 and sim.eco.q > 0:
            print("ATTENTION A LA PRECISION DE L'EQUILIBRE")
            prod = [sim.Q_demand[i, 1, 1] + sim.Q_demand[i, 1, 0] for i in range(len(sim.Q_demand))]
            print("last prod", prod[-1])
            print("last price", sim.prices[-1][0])
    else:
        print("MORE THAN ONE EQUILIBRIUM")
        p_eq_0 = sim.eco.p_eq[0]
        g_eq_0 = sim.eco.g_eq[0]
        p_eq_1 = sim.eco.p_eq[1]
        g_eq_1 = sim.eco.g_eq[1]
        if sim.eco.b != 1 and sim.eco.q > 0:
            print("ATTENTION A LA PRECISION DE L'EQUILIBRE")
            prod = [sim.Q_demand[i, 1, 1] + sim.Q_demand[i, 1, 0] for i in range(len(sim.Q_demand))]
            print("last prod", prod[-1])
            print("last price", sim.prices[-1][0])

    return p_eq_0, g_eq_0

# %%
### Plots equilibrium values
def plot_prices_eq(sim, p_eq_0, scenario):
    """
    Function used for plotting the successive prices in the economy and their
    equilibrium value.
    :param sim: Dynamics object
    :param p_eq_O: array containing a single float, the economy's eq price
    :param scenario: string, describes the set of parameters used in the sim.
    :return: the saved plot
    """
    ### Prices
    fig, ax = plt.subplots()
    ax.set_title("Prices of the firm's production")
    ax.set_xlabel('Time')
    ax.set_ylabel('P1')
    ax.plot(sim.prices[1:-1])
    if p_eq_0 >= 0:
        plt.axhline(y=p_eq_0, linewidth=1.3, alpha=1, color="green", label="p=p_eq")
    #plt.axhline(y=sim.p_eq_1,linewidth=1.3, alpha=1, color="red", label="p=p_eq")
    #plt.xscale("linear")
    ax.set_yscale("log")
    #plt.grid(True)
    file = scenario+"_prices.png"
    fig.savefig(file)

def plot_production_eq(sim, g_eq_0, scenario):
    """
    Function used for plotting the successive production amounts in the economy
    and their equilibrium value.
    :param sim: Dynamics object
    :param g_eq_O: array containing a single float, the economy's eq production
    :param scenario: string, describes the set of parameters used in the sim.
    :return: the saved plot
    """
    ### Production
    fig, ax = plt.subplots()
    ax.set_title("Production")
    ax.set_xlabel('Time')
    ax.set_ylabel("P")
    prod = [sim.Q_demand[i, 1, 1]+sim.Q_demand[i, 1, 0] for i in range(len(sim.Q_demand))]
    ax.plot(prod[1 : -1])
    if g_eq_0 >= 0:
        plt.axhline(y=g_eq_0*sim.eco.firms.z, linewidth=1.3, alpha=1,
                    color="green", label="prod=prod_eq")
    #plt.axhline(y=g_eq_1*sim.eco.firms.z,linewidth=1.3, alpha=1, color="red", label="prod=prod_eq")
    #plt.xscale("linear")s
    ax.set_yscale("log")
    #ax.set_ylim(0,float(max(prod))+100)
    #plt.grid(True)
    file = scenario+"_prods.png"
    fig.savefig(file)

# %%
