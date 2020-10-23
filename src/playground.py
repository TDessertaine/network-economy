#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[55]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Many "phase diagrams": 2D

#values_alpha=[0.3]
#values_alpha_p = [0.05]
#values_beta = [0.1, 0.6, 0.8]
#values_beta_p = [0.05]
#values_w = [0.1, 0.3]

alpha=0.3
alpha_p=0.05
w=0.1

beta=0.1
beta_p=0.05

b=1.0
q=0.0
pert = 0

values_p=[0.051077]
values_g=[np.array([0.4])]


for p_init in values_p:
    for g_init in values_g:
        sim_args = simulations.variables_simulation(alpha, alpha_p, beta, beta_p, w, q, b, p_init, g_init, pert)
        sim = simulations.simulation(**sim_args)
        p_eq_0,g_eq_0=simulations.compute_equilibrium(sim)
        prod = [sim.eco.production_function(sim.Q_real[i, :, 1]) for i in range(len(sim.Q_real))]
        print(p_init, g_init)
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel(r"$p(t)$")
        ax.set_ylabel(r"$\gamma (t)$")
        ax.set_ylim(0, 0.0001)
        scenario=r"$\alpha$="+str(alpha)+r"_$\alpha'$="+str(alpha_p)+r"_$\beta$="+str(beta)+r"_$\beta'$="+str(beta_p)+"_w="+str(w)
        ax.set_title(scenario)

        ax.plot(sim.prices[1:-1],prod[1:-1])
        plt.show()


# In[33]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits import mplot3d
# Many "phase diagrams": 3D
alpha=0.3
alpha_p=0.05
w=0.1

beta=0.1
beta_p=0.05

b=1.0
q=0.0
pert = 0

values_p=[0.051077]
values_g=[np.array([0.4])]

stocks=[sim.stocks[i][0] for i in range(len(sim.stocks))]
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel(r"$p(t)$")
ax.set_ylabel(r"$\gamma (t)$")
ax.set_zlabel(r"stocks")
ax.plot3D(sim.prices[1:-1],prod[1:-1],stocks[1:-1])

plt.show()


# In[32]:


stocks


# In[6]:


### Prices
get_ipython().run_line_magic('matplotlib', 'notebook')
fig, ax = plt.subplots()
ax.set_title("Prices of the firm's production")
ax.set_xlabel('Time')
ax.set_ylabel('P1')

ax.plot(sim.prices[1:-1])

plt.axhline(y=p_eq_0,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
#plt.axhline(y=p_eq_1,linewidth=1.3, alpha=1, color="red", label="p=p_eq")
#plt.xscale("linear")
#plt.axhline(y=1.5, linewidth=1.3, alpha=1, color="red", label="p=p_eq")
ax.set_yscale("log")

#plt.grid(True)
plt.show()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')
### Production
fig, ax = plt.subplots()
ax.set_title("Production")
ax.set_xlabel('Time')
ax.set_ylabel("P")

prod=[sim.eco.production_function(sim.Q_real[i, :, 1]) for i in range(len(sim.Q_real))]

ax.plot(prod[1:-1])
ax.plot(sim.prices[1:-1])
plt.axhline(y=g_eq_0,linewidth=1.3, alpha=1, color="green", label="prod=prod_eq")
#plt.axhline(y=sim.eco.production_function(sim.Q_real[-2, :, 1]),linewidth=1.3, alpha=1, color="red", label="prod=last prod value")

#plt.xscale("linear")s
ax.set_yscale("log")
#ax.set_ylim(0,float(max(prod))+100)
#plt.grid(True)
plt.show()

#directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_08_20_PerturbationsEquilibres_b<1_q=0/"+str(compteur)+"prods_b="+str(econ_args["b"])+"_q="+str(econ_args["q"])+"_"+state+".png"                
#fig.savefig(directoire)


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
t_diff=[]
for t in range(1,1000):
    t_diff.append(np.amax(sim.prices[-1-t-10:-1-t])-np.amin(sim.prices[-1-t-10:-1-t]))
df_t_diff = pd.DataFrame(t_diff[::-1])
print(df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0])
plt.plot(t_diff)


# In[72]:


### Cas témoin: random data

import statsmodels
from statsmodels import stats
from statsmodels.stats import stattools

random_data=[rd.random() for i in range(100)]
absciss = [i for i in range(100)]

print(statsmodels.stats.stattools.durbin_watson(random_data))
plt.plot(absciss, random_data)
plt.show()
lags, c, line, b = plt.acorr(random_data, maxlags=20)
peaks, peaks_properties=scipy.signal.find_peaks(c)
for i in peaks:
    print("longueur période", lags[i], "hauteur pic", c[i])
    plt.axvline(lags[i], color="green")
plt.show()


# In[70]:


### Essai sur modèle 

lags, c, line, b = plt.acorr(t_diff, maxlags=500)
peaks, peaks_properties=scipy.signal.find_peaks(c)
for i in peaks:
    print("longueur période", lags[i], "hauteur pic", c[i])
    plt.axvline(lags[i], color="green")
plt.show()


# In[74]:


get_ipython().run_line_magic('matplotlib', 'notebook')
freq, dft = scipy.signal.periodogram(c, fs=1000)
plt.semilogy(freq, dft)
plt.axvline(freq[np.where(dft==max(dft))[0][0]], color="green")
plt.show()
np.where(dft==max(dft))[0][0]
freq[np.where(dft==max(dft))[0][0]]


# In[66]:





# In[ ]:




