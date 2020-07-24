import numpy as np
import random

from dynamics import Dynamics as dyn
from economy import Economy as eco

import firms 
import household


from graphics_class import PlotlyDynamics as plotdyn
import graphics_df as gd

import plotly.graph_objs as go
import tqdm as tqdm
import plotly.offline as plo

#%% CREATION ECONOMIE

# Dimensions 
n=100
d=7
t_max=10

# Variables statiques Firms
z=np.ones(n)
sigma=np.random.uniform(0,1,n)
alpha=0.1
alpha_p=0.01
beta=0.05
beta_p=0.08
a0=np.ones(n)*0.5
j0=np.ones(n)
w=1
q=0
b=1
# Variables statiques Household
labour=10
theta=np.ones(n)/n
gamma=1
phi=1

econ_args = {
        'n':100,
        'd':7,
        'netstring':'regular',
        'directed':True,
        'j0':np.ones(100),
        'a0':np.ones(100)*0.5,
        'q':0,
        'b':1
        }

house_args = {
        'labour':10,
        'theta':np.ones(econ_args['n']) / econ_args['n'],
        'gamma':1
        }

# Création objet classe Economy
economie=eco(**econ_args)

# Initialisations demandées à Economy
economie.init_house(labour=labour, theta=theta, gamma=gamma, phi=phi)
economie.init_firms(z=z,sigma=sigma, alpha=alpha, alpha_p=alpha_p, beta=beta, beta_p=beta_p, w=w)

economie.set_quantities()




# Création de l'objet dynamique
sim = dyn(t_max=10,e=economie)
#%% PROBLEMES 

pbm_lamb_a=np.hstack((np.array([economie.j0]).T, economie.j))

## Error raised :
# ValueError: all the input arrays must have same number of dimensions


#%% SIMULATION

def InitialisationVecteur(nby,min,max):
    return np.array([random.randint(min, max) for i in range(nby)])

#Conditions initiales
p0=np.random.uniform(1,2,n)
w0=1
g0=np.random.uniform(1,2,n)
s0=np.random.uniform(0,0,n)
t1=np.random.uniform(1,2,n)
B0=random.randint(1,10)

dictionnaire={
        'p0':np.random.uniform(1,2,n),
        'w0':1,
        'g0':np.random.uniform(1,2,n),
        's0':np.random.uniform(0,0,n),
        't1':np.random.uniform(1,2,n),
        'B0':random.randint(1,10)
        }

# Dynamique
sim.discrete_dynamics(**dictionnaire)


#%% PLOT FROM CLASSES DIRECTLY

dic_plots={"dyn":sim,
           "k":int(n/5)+1}

graphes_dyn=plotdyn(sim,k=3)
 

