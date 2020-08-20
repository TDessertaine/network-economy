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

n=1

# Variables statiques Firms

firms_args={
        "z":np.random.uniform(5,5,1),
        "sigma":np.random.uniform(0,1,1),
        "alpha":0.1,
        "alpha_p":0.08,
        "beta":0.2,
        "beta_p":0.05,
        "w":0.02
        }

# Variables statiques Household

house_args = {
        'labour':10,
        'theta':np.ones(1),
        'gamma':1,
        "phi":1
        }


# Variables statiques Economiy
econ_args = {
        'n':1,
        'd':1,
        'netstring':'regular',
        'directed':True,
        'j0':np.array([3]),
        'j1':np.array([5]),
        'a0':np.ones(1)*0.5,
        'q':0.5,
        'b':0.95
        }


# Variables statiques Dynamics
p_init=5
g_init=np.array([3])

dyn_args={
        'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':g_init,
        's0':np.random.uniform(0,0,econ_args['n']),
        't1':g_init,
        'B0':random.randint(0,0)
        }



# %% %% SIMULATION
# Création objet classe Economy
economie=eco(**econ_args)

# Initialisations demandées à Economy
economie.init_house(**house_args)
economie.init_firms(**firms_args)

economie.set_quantities()


# Création de l'objet dynamique
sim = dyn(t_max=10000,e=economie)


# Dynamique
sim.discrete_dynamics(**dyn_args)


#%%

# EQUILIBRIUM


#%%  COMPUTE EQUILIBRIUM

sim.eco.compute_eq()
print("P_EQ", sim.eco.p_eq)
print("G_EQ",sim.eco.g_eq)

p_eq_dis_0=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
g_eq_0=sim.eco.g_eq[0]

p_eq_dis_1=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
g_eq_1=sim.eco.g_eq[1]

if sim.eco.b!=1 and sim.eco.q>0:
    print("ATTENTION A LA PRECISION DE L'EQUILIBRE")
    print(sim.prices[-1][0])
    prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]
    print(prod[-1])

#%% PLOT EQUILIBRIUM ON PRICES 

import matplotlib as mpl
import matplotlib.pyplot as plt

### Prices
plt.title("Prices of the firm's production")
plt.xlabel('Time')
plt.ylabel('P1')

plt.plot(sim.prices[1:-1])

plt.axhline(y=p_eq[0],linewidth=1.3, alpha=1, color="green", label="p=p_eq")
plt.axhline(y=p_eq[1],linewidth=1.3, alpha=1, color="red", label="p=p_eq")
#plt.xscale("linear")
plt.yscale("log")

#plt.grid(True)
plt.show()

#MKDIR NOT DONE: plt.savefig("/mnt/research-live/user/cboissel/network-economy/OneFirmCase_Images_v1/prices.png")


#%% PLOT EQUILIBRIUM ON PRODUCTION 

import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
### Production
fig, ax = plt.subplots()
ax.set_title("Production")
ax.set_xlabel('Time')
ax.set_ylabel("P")

prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]

ax.plot(prod[1:-1])

plt.axhline(y=sim.eco.g_eq[0]*sim.eco.firms.z,linewidth=1.3, alpha=1, color="green", label="prod=prod_eq")

#plt.xscale("linear")
ax.set_yscale("log")
ax.set_ylim(0,float(max(sim.labour))+100)
#plt.grid(True)
plt.show()

