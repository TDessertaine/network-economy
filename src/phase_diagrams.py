#!/usr/bin/env python
# coding: utf-8
# %%
"""
Created on Tue Sep 29 02:22:16 2020
@author: boisselcamille
Functions needed to set up phase diagrams from a simulation
with initial price just perturbed from equilibrium value:
- classifies type of behaviour
- plots phase diagrams for different values for the timescales
"""


# %%
import random as rd 
import os
import re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import colors
import scipy
from scipy import signal
import simulations
import stability_diagrams as sd


# %%
# Detect 5 different kinds of behaviour 
def fisher_test(sim):
    prices=sim.prices.T[0][-1000:-1]
    freq, dft = scipy.signal.periodogram(prices, fs=1)
    q = int((len(prices)-1)/2)
    stat = max(dft)/np.sum(dft)
    b=int(1/stat)
    p_value = 1 - np.sum([((-1)**j)*scipy.special.binom(q, j)*((1-j*stat)**(q-1)) for j in range(b+1)])
    return p_value
    
def detect_periodicity(sim):
    """
    Function used to determine if prices are periodic.
    :param sim: Dynamics object
    :return: True if periodic, False otherwise
    """  
    return fisher_test(sim) < 0.001

def detect_conv(sim):
    """
    Function used to determine if prices converge.
    :param sim: Dynamics object
    :return: True if prices converge, False otherwise
    """    
    return sd.rolling_diff(sim)

def detect_div(sim):
    """
    Function used to determine if prices diverge.
    :param sim: Dynamics object
    :return: True if prices diverge, False otherwise
    """    
    if np.isnan(sim.prices.T[0][-2]):
        return True
    else:
        t_diff=[]
        for t in range(1,500):
            t_diff.append(np.amax(sim.prices[-1-t-10:-1-t])-np.amin(sim.prices[-500:-1]))
        df_t_diff = pd.DataFrame(t_diff[::-1])
        if df_t_diff.apply(lambda x: x.is_monotonic_increasing)[0] and df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0]:
            return False
        else:
            return df_t_diff.apply(lambda x: x.is_monotonic_increasing)[0]
    
def detect_crises(sim):
    """
    Function used to determine if prices converge or diverge.
    :param sim: Dynamics object
    :return: True in converges, False otherwise
    """   
    fact_norm=float(max(sim.prices[-1000:-1]))
    peaks_minus, peaks_properties_minus=scipy.signal.find_peaks(-sim.prices.T[0][-1000:-1]/fact_norm)
    peaks_plus, peaks_properties_plus=scipy.signal.find_peaks(sim.prices.T[0][-1000:-1]/fact_norm)
    return np.average(sim.prices.T[0][-1000:-1][peaks_plus])/np.average(sim.prices.T[0][-1000:-1][peaks_minus]) > 10e2
    


# %%
def phase_classification(sim):
    if detect_div(sim):
        return 0
    else:
        if detect_conv(sim): # TODO : peut-être insérer différentiation eq inflationnaire / eq compétitif
            return 1
        else:
            if detect_periodicity(sim):
                if detect_crises(sim):
                    return 3
                else:
                    return 2

            else:
                return 4
        
