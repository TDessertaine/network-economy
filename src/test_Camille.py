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


# %%

# SIMULATION

# %% CREATION ECONOMIE

# Variables statiques Firms
firms_args={
        "z":np.random.uniform(5,5,1),
        "sigma":np.random.uniform(0,1,1),
        "alpha":0.1,
        "alpha_p":0.1,
        "beta":0.1,
        "beta_p":0.05,
        "w":0.05
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
        'j0':np.array([5]),
        'j1':np.array([2]),
        'a0':np.ones(1)*0.5,
        'q':0,
        'b':1
        }

# Variables statiques Dynamics
p_init=5
g_init=np.array([3])

dyn_args={
        'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':g_init,
        's0':np.random.uniform(0,0,1),
        't1':g_init,
        'B0':random.randint(0,0)
        }


# Variables de suivi des perturbations pour un dossier donné (last: 20_08_2020)
state="nd"
compteur=5

# %%%% SIMULATION
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


# %%

# EQUILIBRIUM


# %% COMPUTE EQUILIBRIUM

sim.eco.compute_eq()
print("P_EQ", sim.eco.p_eq)
print("G_EQ",sim.eco.g_eq)

p_eq_0=sim.eco.p_eq[0]
g_eq_0=sim.eco.g_eq[0]
p_eq_dis_0=sim.eco.p_eq[0] + random.uniform(-10**(-5),10**(-5))
g_eq_dis_0=sim.eco.g_eq[0] + random.uniform(-10**(-5),10**(-5))

#g_eq_1=sim.eco.g_eq[1]
#p_eq_1=sim.eco.p_eq[1]
#p_eq_dis_1=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
#g_eq_dis_1=sim.eco.g_eq[1] + random.uniform(-10**(-5),10**(-5))


if sim.eco.b!=1 and sim.eco.q>0:
    print("ATTENTION A LA PRECISION DE L'EQUILIBRE")
    print(sim.prices[-1][0])
    prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]
    print(prod[-1])

# %% PLOT EQUILIBRIUM ON PRICES

import matplotlib as mpl
import matplotlib.pyplot as plt

### Prices
fig, ax = plt.subplots()
ax.set_title("Prices of the firm's production")
ax.set_xlabel('Time')
ax.set_ylabel('P1')

ax.plot(sim.prices[1:-1])

plt.axhline(y=p_eq_0,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
plt.axhline(y=sim.p_eq_1,linewidth=1.3, alpha=1, color="red", label="p=p_eq")
#plt.xscale("linear")
ax.set_yscale("log")

#plt.grid(True)
plt.show()

#MKDIR NOT DONE: plt.savefig("/mnt/research-live/user/cboissel/network-economy/OneFirmCase_Images_v1/prices.png")
directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_08_20_PerturbationsEquilibres_b<1_q=0/"+str(compteur)+"prices_b="+str(econ_args["b"])+"_q="+str(econ_args["q"])+"_"+state+".png"                

#fig.savefig(directoire)

# %% PLOT EQUILIBRIUM ON PRODUCTION

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

plt.axhline(y=g_eq_0*sim.eco.firms.z,linewidth=1.3, alpha=1, color="green", label="prod=prod_eq")
#plt.axhline(y=g_eq_1*sim.eco.firms.z,linewidth=1.3, alpha=1, color="red", label="prod=prod_eq")

#plt.xscale("linear")s
ax.set_yscale("log")
#ax.set_ylim(0,float(max(prod))+100)
#plt.grid(True)
plt.show()

directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_08_20_PerturbationsEquilibres_b<1_q=0/"+str(compteur)+"prods_b="+str(econ_args["b"])+"_q="+str(econ_args["q"])+"_"+state+".png"                
#fig.savefig(directoire)


# %% PLOT EQUILIBRE INFLATIONNAIRE

### Coefficient d'update des prix
update_labour=[np.exp(-sim.eco.firms.w*(sim.labour[i-1]-sim.Q_demand[i-1][0,1])/sim.labour[i-1]) for i in range(1,len(sim.Q_real))]

profits=[sim.prices[i]*sim.Q_real[i][1,0]-sim.Q_real[i][0,1] for i in range(1,len(sim.Q_real))]

cashflow=[sim.prices[i]*(sim.Q_real[i][1,0]+2*sim.Q_real[i][1,1])+sim.Q_real[i][0,1] for i in range(1,len(sim.Q_real))]

def production_function(simulation, Q):

    if simulation.eco.q == 0:
        return np.power(np.min(np.ma.masked_invalid(np.divide(Q, simulation.eco.j_a))),
                        simulation.eco.b)
    elif simulation.eco.q == np.inf:
        return np.power(np.prod(np.power(np.ma.masked_invalid(np.divide(Q, simulation.eco.j_a)),
                                         simulation.eco.a_a)),
                        simulation.eco.b)
    else:
        return np.power(np.sum(np.ma.masked_invalid(simulation.eco.a_a * np.power(simulation.eco.j_a, 1. / simulation.eco.q)
                                                    * np.power(Q, -1. / simulation.eco.q))),
                        - simulation.eco.b * simulation.eco.q)
            
balance=[sim.eco.firms.z*production_function(simulation=sim,Q=[sim.Q_demand[i-1][0,1],sim.Q_demand[i-1][1,1]])+sim.stocks[i]-sim.Q_demand[i][1,0]+sim.Q_demand[i][1,1] for i in range(1,len(sim.Q_real))]

biens_circ=[sim.stocks[i]+sim.eco.firms.z*production_function(simulation=sim,Q=[sim.Q_demand[i-1][0,1],sim.Q_demand[i-1][1,1]])+sim.Q_demand[i][1,0]+sim.Q_demand[i][1,1] for i in range(1,len(sim.Q_real))]

update_prix=[np.exp(-sim.eco.firms.alpha*balance[i]/biens_circ[i]-sim.eco.firms.alpha_p*profits[i]/cashflow[i]) for i in range(len(sim.Q_real)-1)]

update_prix_rescale=[update_prix[i]/update_labour[i] for i in range(len(update_prix))]

#plt.plot(balance)
plt.plot(update_prix_rescale)


couts_rel=[sim.prices[i]*sim.Q_real[i,1,1]-sim.Q_real[i,1,0] for i in range(1,len(sim.Q_real))]

### Coefficient d'update des niveaux de consommation





# %% VERIFICATION HYPOTHESE SATISFACTION DU MENAGE IN EQ INFL vs IN EQ GENERAL

utilite=[sim.eco.house.utility(sim.Q_real[i][1,0],sim.Q_real[i][0,1]) for i in range(len(sim.Q_real))]

import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
### Production
fig, ax = plt.subplots()
ax.set_title("Household's utility")
ax.set_xlabel('Time')
ax.set_ylabel("Utility")

ax.plot(utilite[1:-1])


#plt.xscale("linear")
ax.set_yscale("log")

#plt.grid(True)
plt.show()

# %% EXPERIENCE PIB

pib_prod=[]

pib_revenus=[]

pib_demande=[]
