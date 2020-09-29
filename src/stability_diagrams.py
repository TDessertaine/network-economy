#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Sep 29 02:22:16 2020

@author: boisselcamille
"""

"""
Functions needed to set up stability diagrams:
- classifies type of behaviour
- plots stability diagrams for different values for the timescales
"""

# %%
import random as rd 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import simulations

# %%
### Classify Equilibrium
# Typology
def classify_p_inf(sim, p_eq_0, threshold=1e-6):
    """
    Function used for classifying the long-term behaviour of the prices in  a
    single simulation.
    :param sim: Dynamics object
    :param p_eq_O: array containing a single float, the economy's eq price
    :param threshold: float, threshold for the function's precision
    :return: string, prices' behaviour
    """
    std_diff = np.std(sim.prices[-101:-1]-sim.prices[-102:-2])
    if std_diff <= threshold:
        if np.abs(sim.prices[-10]*(sim.Q_real[-10][1, 0])-sim.Q_real[-10][0, 1]) <= threshold:
            print(p_eq_0)
            p_inf = "conv_eq"
        else:
            print(sim.prices[-10]*(sim.Q_real[-10][1, 0])-sim.Q_real[-10][0, 1])
            p_inf = "conv_infl"
    elif np.mean(np.log(std_diff)) >= 10:
        p_inf = "div_exp"
    else:
        p_inf = "div"
    return p_inf

# Exponent of the enxponential
def compute_exp_exponent(sim, t_max=500):
    """
    Function used for computing the k value for stability diagrams
    (description to come).
    :param sim: Dynamics object
    :param t_max: float, the simulation's duration
    :return: k value
    """
    return np.diff(np.array([float(i) for i in np.log(sim.prices[-101:-1])]),
                   n=1)/np.diff(np.array(range(t_max-100, t_max)))[-1]

# %%
def rolling_diff(sim, threshold=1e-6):
    """
    Function used for classifying the long-term behaviour of the prices in  a
    single simulation: rolling diff version.
    :param sim: Dynamics object
    :param threshold: float, threshold for the function's precision
    :return: Bool, for "prices converge" statement
    """
    t_diff=[]
    for t in range(1,10):
        t_diff.append(np.amax(sim.prices[-1-10*t:-1])-np.amin(sim.prices[-1-10*t:-1]))
    df_t_diff = pd.DataFrame(t_diff[::-1])
    return df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0]
    
def comparison(sim_1, var_sim_1, t_max):
    """
    Function used for classifying the long-term behaviour of the prices in  a
    single simulation: comparing two simulations version.
    :param sim_1: Dynamics object, the original simulation
    :param sim_2: Dynamics object, the simulation sim_1 is compared with
    :param threshold: float, threshold for the function's precision
    :return: string, prices' behaviour
    """
    sim_2_args = simulations.variables_simulation(sim_1.eco.firms.alpha, 
                                     sim_1.eco.firms.alpha_p, 
                                     sim_1.eco.firms.beta, 
                                     sim_1.eco.firms.beta_p, 
                                     sim_1.eco.firms.w, 
                                     sim_1.eco.q, sim_1.eco.b, 
                                     pert=rd.uniform(-1e-6,1e-6))
    sim_2 = simulations.simulation(**sim_2_args)
    
    diff_simus = sim_2.prices[-101:-1]-sim_1.prices[-101:-1]
    df_diff_simus = pd.DataFrame(diff_simus)
    if df_diff_simus.apply(lambda x: x.is_monotonic_decreasing)[0]:
        exponents = np.diff(np.array([float(i) for i in np.log(sim_1.prices[-101:-1]-sim_1.prices[-2])]),
                   n=1)/np.diff(np.array(range(t_max-100, t_max)))
        return exponents[-2]
    else:
        sim_3_args = simulations.variables_simulation(0, 
                                     sim_1.eco.firms.alpha_p, 
                                     sim_1.eco.firms.beta, 
                                     sim_1.eco.firms.beta_p, 
                                     sim_1.eco.firms.w, 
                                     sim_1.eco.q, sim_1.eco.b, 
                                     pert=rd.uniform(-1e-6,1e-6))
        exponents = np.diff(np.array([float(i) for i in np.log(sim_1.prices[-101:-1])]),
                   n=1)/np.diff(np.array(range(t_max-100, t_max)))
        return exponents[-2]    

# %%
### Plot Stability Diagrams (colormap & scatterplot)
def plot_stabilitydiagramm_be(data_diagramme_x, data_diagramme_y,
                              data_diagramme_be, title, directoire):
    """
    Function used for plotting the stability diagrams.
    :param data_diagramme_x: timescale x values
    :param data_diagramme_y: timescale y values
    :param data_diagramme_be: k value for given (x,y)
    :param title: string, describes the set of parameters used in the sim.
    :param directoire: string, file used to save figures.
    :return: the saved plot
    """
    fig, ax = plt.subplots()
    ax.scatter(data_diagramme_x, data_diagramme_y, c=data_diagramme_be)
    ax.set_title(title)
    ax.set_xlabel("beta_p")
    ax.set_ylabel("beta")
    ax.set_xlim(-0.05, 1.05)
    im = ax.scatter(data_diagramme_x, data_diagramme_y, c=data_diagramme_be)
    fig.colorbar(im, ax=ax)   # pour classification
    fig.savefig(directoire+"/"+title+".png")

def plot_stabilitydiagramm_exp(data_diagramme_x, data_diagramme_y,
                               data_diagramme_be, title, values, directoire):
    """
    Function used for plotting the stability diagrams.
    :param data_diagramme_x: timescale x values
    :param data_diagramme_y: timescale y values
    :param data_diagramme_be: prices' long term behaviour for given (x,y)
    :param title: string, describes the set of parameters used in the sim.
    :param directoire: string, file used to save figures.
    :return: the saved plot
    """
    coordonnees = {}
    for i in range(len(values)):
        coordonnees[values[i]] = i
    data_slope = np.zeros((len(values), len(values)))
    for i in range(len(values)**2):
        data_slope[coordonnees[data_diagramme_y[i]],
                   coordonnees[data_diagramme_x[i]]] = data_diagramme_be[i]
    fig, ax = plt.subplots()
    y, x = np.mgrid[0:11, 0:11]/10
    im = ax.pcolor(x, y, data_slope, cmap='RdBu_r',
                   norm=colors.SymLogNorm(linthresh=1e-9, linscale=1))
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title("k")
    ax.set_title(title)
    ax.set_xlabel("alpha_p")
    ax.set_ylabel("beta_p")
    fig.show()
    fig.savefig(directoire+"/"+title+".png")

# %%
### Creates a GIF with Stability Diagrams
def serialization_stabilitydiagrams(i, values, behaviour, directoire):
    """
    Function used for plotting the stability diagrams.
    :param i: index of the stability diagram plot
    :param values: timescales values selected
    :param behaviour: prices' behaviour being studied
    :param directoire: string, file used to save figures.
    :return: a saved plot
    """
    values_ext = values*3
    alpha = values_ext[int(i/9)]
    alpha_p = values_ext[int(i/3)]
    w = values_ext[i]
    data_diagramme_x = []
    data_diagramme_y = []
    data_diagramme_be = []
    for key in behaviour:
        if "alpha="+str(alpha)+"_" in key and "alpha_p="+str(alpha_p)+"_" in key and re.search("w="+str(w)+"$", key):
            beta_p = float(re.findall(r'beta_p=(\d+\.?\d*)_', key)[0])
            beta = float(re.findall(r'beta=(\d+\.?\d*)_', key)[0])
            data_diagramme_x.append(beta_p)
            data_diagramme_y.append(beta)
            data_diagramme_be.append(behaviour[key])
    nb_be = len(set(data_diagramme_be))
    title = "Stability Diagram. Types of behaviour:"+str(nb_be)+". \n alpha="+str(alpha)+"_"+"alpha_p="+str(alpha_p)+"_"+"w="+str(w)
    plot_stabilitydiagramm_exp(data_diagramme_x, data_diagramme_y, data_diagramme_be, title, values, directoire)


# %%
