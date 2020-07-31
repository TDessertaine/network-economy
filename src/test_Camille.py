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



#import tqdm as tqdm


# %% CREATION ECONOMIE

# Dimensions 
n=1
d=1


# Variables statiques Firms
z=np.ones(n)
sigma=np.random.uniform(0,1,1)
alpha=0.2
alpha_p=0.6
beta=0.5
beta_p=0.4
a0=np.ones(n)*0.5
j0=np.ones(n)
w=1
q=0
b=1
# Variables statiques Household
labour=100
theta=np.ones(n)/n
gamma=1
phi=1

econ_args = {
        'n':1,
        'd':1,
        'netstring':'regular',
        'directed':True,
        'j0':np.ones(1),
        'j1':np.ones(1),
        'a0':np.ones(1)*0.5,
        'q':1,
        'b':0.1
        }

house_args = {
        'labour':5,
        'theta':np.ones(econ_args['n']) / econ_args['n'],
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
sim = dyn(t_max=200,e=economie)

# %% SIMULATION

def InitialisationVecteur(nby,min,max):
    return np.array([random.randint(min, max) for i in range(nby)])

#Conditions initiales
dictionnaire={
        'p0':np.array([50]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':np.random.uniform(15,20,econ_args['n']),
        's0':np.random.uniform(3,4,econ_args['n']),
        't1':np.random.uniform(1,1,econ_args['n']),
        'B0':random.randint(2,3)
        }

# Dynamique
sim.discrete_dynamics(**dictionnaire)
#sim.prices
#sim.labour

# %%

#economie.compute_eq()
#economie.v
#economie.m_cal

# %% PLOT
import matplotlib as mpl
mpl.use('WebAgg')
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
sim.Q_real[:,1,0]

# %%
import matplotlib as mpl
mpl.use('WebAgg')
import matplotlib.pyplot as plt
### Production
plt.title("Labour supply")
plt.xlabel('Time')
plt.ylabel('P1')

plt.plot(sim.Q_real[:,1,1])

#plt.xscale("linear")
plt.yscale("log")
plt.ylim(0,float(max(sim.labour))+100)
#plt.grid(True)
plt.show()

# %% DEBUG
sim.labour

