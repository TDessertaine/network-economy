# -*- coding: utf-8 -*-
# %%
import os,sys
#sys.path.append('/mnt/research-live/user/cboissel/network-economy/src/')
sys.path.append('/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/network-economy/src')
import numpy as np
import random

from dynamics import Dynamics as dyn
from economy import Economy as eco

import firms 
import household


#%%

# SIMULATION

# %% CREATION ECONOMIE



# Variables statiques Firms
z=np.random.uniform(3,3,1)
sigma=np.random.uniform(0,1,1)
alpha=0.1
alpha_p=0.08
beta=0.2
beta_p=0.05
w=0.02


# Variables statiques Household
labour=50
theta=np.ones(n)/n
gamma=1
phi=1

# Variables statiques Economiy
econ_args = {
        'n':1,
        'd':1,
        'netstring':'regular',
        'directed':True,
        'j0':np.array([3]),
        'j1':np.array([1]),
        'a0':np.ones(1)*0.5,
        'q':2,
        'b':1
        }

house_args = {
        'labour':50,
        'theta':np.ones(1),
        'gamma':1
        }


# %%
# Création objet classe Economy
economie=eco(**econ_args)

# Initialisations demandées à Economy
economie.init_house(labour=labour, theta=theta, gamma=gamma, phi=phi)
economie.init_firms(z=z,sigma=sigma, alpha=alpha, alpha_p=alpha_p, beta=beta, beta_p=beta_p, w=w)

economie.set_quantities()




# Création de l'objet dynamique
sim = dyn(t_max=1000,e=economie)

# %% SIMULATION

#Conditions initiales
dictionnaire={
        'p0':np.array([5]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':np.random.uniform(2,3,econ_args['n']),
        's0':np.random.uniform(0,0,econ_args['n']),
        't1':np.random.uniform(2,3,econ_args['n']),
        'B0':random.randint(0,0)
        }

# Dynamique
sim.discrete_dynamics(**dictionnaire)
#sim.prices
#sim.labour

# %%

# PLOTS 

# %% PLOT
import matplotlib as mpl
#mpl.use('WebAgg')
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg')
# %matplotlib inline 
### Prices
plt.title("Prices of the firm's production")
plt.xlabel('Time')
plt.ylabel('P1')

plt.plot(sim.prices)

#plt.xscale("linear")
plt.yscale("log")
plt.ylim(1,float(max(sim.prices))+10000)
#plt.grid(True)
plt.show()

#MKDIR NOT DONE: plt.savefig("/mnt/research-live/user/cboissel/network-economy/OneFirmCase_Images_v1/prices.png")


# %%
import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
### Production
fig, ax = plt.subplots()
ax.set_title("Labour supply")
ax.set_xlabel('Time')
ax.set_ylabel('P1')

ax.plot(sim.Q_demand[:,1,1])

#plt.xscale("linear")
ax.set_yscale("log")
ax.set_ylim(0,float(max(sim.labour))+100)
#plt.grid(True)
plt.show()


#%%

# EQUILIBRIUM


#%%  COMPUTE EQUILIBRIUM

sim.eco.compute_eq()
print("P_EQ", sim.eco.p_eq)
print("G_EQ",sim.eco.g_eq)

#%% PLOT EQUILIBRIUM ON PRICES 

import matplotlib as mpl
import matplotlib.pyplot as plt

### Prices
plt.title("Prices of the firm's production")
plt.xlabel('Time')
plt.ylabel('P1')

plt.plot(sim.prices)

plt.axhline(y=sim.eco.p_eq[0],linewidth=1.3, alpha=1, color="green", label="p=p_eq")
plt.axhline(y=sim.eco.p_eq[1],linewidth=1.3, alpha=1, color="red", label="p=p_eq")
#plt.xscale("linear")
plt.yscale("log")

#plt.grid(True)
plt.show()

#%% PLOT EQUILIBRIUM ON PRODUCTION 

import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
### Production
fig, ax = plt.subplots()
ax.set_title("Labour supply")
ax.set_xlabel('Time')
ax.set_ylabel("l")

prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]

ax.plot(prod)

plt.axhline(y=sim.eco.g_eq*sim.eco.firms.z,linewidth=1.3, alpha=1, color="green", label="prod=prod_eq")

#plt.xscale("linear")
ax.set_yscale("log")
ax.set_ylim(0,float(max(sim.labour))+100)
#plt.grid(True)
plt.show()

#%% Cas b=1 & q<+inf
x_s=[i/10 for i in range(1000)]
y=[(sim.eco.firms.z*x)**(1/(sim.eco.q+1))-sim.eco.lamb_a[1]*x-sim.eco.lamb_a[0] for x in x_s]
c=[0 for x in x_s]
x=lstsq(y,c)
plt.plot(y[:50])