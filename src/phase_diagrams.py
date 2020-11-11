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
from scipy import optimize
from scipy import stats
import simulations
import stability_diagrams as sd
import numpy.fft


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
    

#%% 
def sin_fit(x_data, y_data):
    """
    Function used to fit sin to the input time series.
    :return: fitted parameters (amp_n, omega_n, phase_n : for n in [1,..., nbr_freq])
    """
    ff = np.fft.fftfreq(len(x_data), 1) 
    Fyy = abs(np.fft.fft(y_data))
    peaks, peaks_properties = scipy.signal.find_peaks(Fyy[1:])
    Fyy_trunc = Fyy[1:int(len(Fyy)/2)+1]
    Fyy_trunc_peaks = Fyy[peaks+1][:int(len(peaks)/2)]
    nbr_freq = min(3,len(Fyy_trunc_peaks))
    guess_freqs = [ff[int(np.where(Fyy_trunc==i)[0])+1] for i in np.sort(Fyy_trunc_peaks)[-nbr_freq:]]
    guess_offset = np.mean(y_data)
    guess_amp = np.max(y_data)-guess_offset
    print(nbr_freq)
    if nbr_freq==0:
        return None 
    elif nbr_freq==1:
        guess = np.array([guess_amp, 2.*np.pi*guess_freqs[0], 0., guess_offset])
        def sin_func(t, A, w, p, c):  
            return A * np.sin(w*t + p) + c
        params, params_covariance = optimize.curve_fit(sin_func, x_data, y_data, p0=guess)
        amp1, omega1, phase1, offset = params
        amp2 = 0 
        omega2 = 0
        phase2 = 0
        amp3 = 0 
        omega3 = 0
        phase3 = 0
    elif nbr_freq==2:
        guess = np.array([guess_amp/2, 2.*np.pi*guess_freqs[0], 0., 
                          guess_amp, 2.*np.pi*guess_freqs[1], 0., guess_offset])
        def sin_func(t, A1, w1, p1, A2, w2, p2, c):  
            return A1 * np.sin(w1*t + p1) + A2 * np.sin(w2*t + p2) + c
        params, params_covariance = optimize.curve_fit(sin_func, x_data, y_data, p0=guess)
        amp1, omega1, phase1, amp2, omega2, phase2, offset = params
        amp3 = 0 
        omega3 = 0
        phase3 = 0
    elif nbr_freq==3:
        guess = np.array([guess_amp/4, 2.*np.pi*guess_freqs[0], 0., 
                          guess_amp/2, 2.*np.pi*guess_freqs[1], 0., 
                          guess_amp, 2.*np.pi*guess_freqs[2], 0., guess_offset])
        def sin_func(t, A1, w1, p1, A2, w2, p2, A3, w3, p3, c):  
            return A1 * np.sin(w1*t + p1) + A2 * np.sin(w2*t + p2) + A3 * np.sin(w3*t + p3) + c
        params, params_covariance = optimize.curve_fit(sin_func, x_data, y_data, p0=guess)
        amp1, omega1, phase1, amp2, omega2, phase2, amp3, omega3, phase3, offset = params
    
    return amp1, omega1, phase1, amp2, omega2, phase2, amp3, omega3, phase3, offset 
    
def fitted_func(x, amp1, omega1, phase1, amp2, omega2, phase2, amp3, omega3, phase3, offset):
    return amp1 * np.sin(omega1 * x + phase1) + amp2 * np.sin(omega2 * x + phase2) + amp3 * np.sin(omega3 * x + phase3) + offset

def fit_test(y_data, fit_data):
    stat, p_value = stats.ks_2samp(y_data, fit_data)
    return p_value

def detect_periodicity_fit(sim):
    x_data= np.array([i for i in range(len(sim.prices)-1000, len(sim.prices)-1)])
    y_data= sim.prices.T[0][-1000:-1]
    amp1, omega1, phase1, amp2, omega2, phase2, amp3, omega3, phase3, offset  = sin_fit(x_data,y_data)
    fit_data = fitted_func(x_data, amp1, omega1, phase1, amp2, omega2, phase2, amp3, omega3, phase3, offset)
    return fit_test(y_data, fit_data) > 0.0001
    

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
        
