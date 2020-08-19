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
z=np.random.uniform(5,5,1)
sigma=np.random.uniform(0,1,1)
alpha=0.1
alpha_p=0.08
beta=0.2
beta_p=0.05
w=0.02


# Variables statiques Household
labour=3
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
        'j1':np.array([5]),
        'a0':np.ones(1)*0.5,
        'q':0.5,
        'b':0.95
        }

house_args = {
        'labour':10,
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
sim = dyn(t_max=10000,e=economie)

# %% SIMULATION
p_init=5
g_init=np.array([3])


#Conditions initiales
dictionnaire={
        'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
        'w0':1,
        'g0':g_init,
        's0':np.random.uniform(0,0,econ_args['n']),
        't1':g_init,
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

plt.plot(sim.prices[1:-1])

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

ax.plot(sim.Q_demand[:,1,1][1:-1])

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

p_eq_dis_0=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
g_eq_0=sim.eco.g_eq[0]

p_eq_dis_1=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
g_eq_1=sim.eco.g_eq[1]

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

#%% Cas b=1 & q<+inf
from scipy.optimize import fsolve

def non_linear_eq_b1_qnonzero(x):
        return ((sim.eco.firms.z*x)**(1/(sim.eco.q+1)))-sim.eco.lamb_a[1]*x-sim.eco.lamb_a[0]

x_s=[i/10 for i in range(1000)]
    
y=[non_linear_eq_b1_qnonzero(x) for x in x_s]
y_abs=[abs(non_linear_eq_b1_qnonzero(x)) for x in x_s]

c=[0 for x in x_s]

initial_guess_peq=10**8

x_eq=y_abs.index(min(y_abs))

x_eq=[]
initial_guess_peq_1=sim.eco.firms.z**(sim.eco.q+1)+sim.eco.lamb_a[1]+sim.eco.lamb_a[0]
initial_guess_peq_2=0
x_eq.append(fsolve(non_linear_eq_b1_qnonzero,initial_guess_peq_1))
x_eq.append(fsolve(non_linear_eq_b1_qnonzero,initial_guess_peq_2))


plt.axvline(x=x_eq[0]*10,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
plt.axvline(x=x_eq[1]*10,linewidth=1.3, alpha=1, color="red", label="p=p_eq")

plt.plot(y_abs[:100])

#%% Cas b!=1 & q<+inf
from scipy.optimize import fsolve 
from scipy.optimize import root
from scipy.optimize import show_options

#show_options(solver="root")
################# First graphic equation

def non_linear_eq_b_q_finites(u):
    return sim.eco.house.kappa+(sim.eco.firms.z**(sim.eco.q*sim.eco.zeta))*sim.eco.lamb_a[1]*u**(sim.eco.coefficient)-sim.eco.firms.z*u
    

x_s=[i/10 for i in range(1,10000)]

points=[non_linear_eq_b_q_finites(j) for j in x_s]
points_abs=[abs(non_linear_eq_b_q_finites(j)) for j in x_s]

x_eq=[]
#self.g_eq=[]
initial_guess_peq_1=0
initial_guess_peq_2=((sim.eco.firms.z**(sim.eco.zeta))/(sim.eco.lamb_a[1]))**(1/(sim.eco.coefficient-1))
x_eq.append(fsolve(non_linear_eq_b_q_finites,initial_guess_peq_1))
x_eq.append(fsolve(non_linear_eq_b_q_finites,initial_guess_peq_2))

#plt.axvline(x=x_eq[0]*10,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
#plt.axvline(x=x_eq[1]*10,linewidth=1.3, alpha=1, color="red", label="p=p_eq")

#plt.axhline(y=0,linewidth=1.3, alpha=1, color="black", label="p=p_eq")

#plt.plot(points)


################# Second graphic equation

p_eq=[]

for term in x_eq:
    initial_guess_v=10
    def non_linear_eq_b_q_finites_2(v):
        """
        Function used to solve the first equilibrium equations in the case 
        b<1 and q<+inf
        :param x: zeta*(b*q+1)/b
        """
        return sim.eco.lamb_a[0]*(sim.eco.firms.z**(-sim.eco.zeta))*(v**(1/(1-sim.eco.b)))+v*sim.eco.lamb_a[1]*(sim.eco.firms.z**(-sim.eco.zeta))-(term**(1-sim.eco.coefficient))
    def non_linear_eq_b_q_finites_2_prime(v):
        return sim.eco.lamb_a[0]*(sim.eco.firms.z**(-sim.eco.zeta))*(1/(1-sim.eco.b))*(v**(sim.eco.b/(1-sim.eco.b)))+(sim.eco.firms.z**(-sim.eco.zeta))*sim.eco.lamb_a[1]
    
    nuage=[non_linear_eq_b_q_finites_2(j) for j in x_s]
    nuage_abs=[abs(i) for i in nuage]
    
    p_eq=fsolve(func=non_linear_eq_b_q_finites_2, x0=initial_guess_v, fprime=non_linear_eq_b_q_finites_2_prime)
    
    p_eq_r=root(fun=non_linear_eq_b_q_finites_2, x0=initial_guess_v, method="diagbroyden")
    
    print(p_eq**(1/(sim.eco.coefficient-1)))
    print(np.power(p_eq_r,(1/(sim.eco.coefficient-1))))
    
    plt.axvline(x=p_eq,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
    plt.axvline(x=nuage_abs.index(min(nuage_abs)),linewidth=1.3, alpha=1, color="yellow", label="p=p_eq")
    plt.axhline(y=0,linewidth=1.3, alpha=1, color="black", label="p=p_eq")
    plt.plot(nuage[:10])
    #p_eq.append((fsolve(non_linear_eq_b_q_finites_2,initial_guess_v))**(1/(sim.eco.coefficient-1)))




# Broyden1 : array(0.95007593)
# lm: FALSE
# Broyden2: array(0.95007593)
# diagbroyden: array(0.95007593)
# fsolve: array([0.95007567])
#
    
#%%%

from scipy.optimize import minimize
p_eq=[]

for term in x_eq:
    initial_guess_v=10
    def non_linear_eq_b_q_finites_2(v):
        """
        Function used to solve the first equilibrium equations in the case 
        b<1 and q<+inf
        :param x: zeta*(b*q+1)/b
        """
        return sim.eco.lamb_a[0]*(sim.eco.firms.z**(-sim.eco.zeta))*(v**(1/(1-sim.eco.b)))+v*sim.eco.lamb_a[1]*(sim.eco.firms.z**(-sim.eco.zeta))-(term**(1-sim.eco.coefficient))
    
    def non_linear_eq_b_q_finites_2_prime(v):
        return sim.eco.lamb_a[0]*(sim.eco.firms.z**(-sim.eco.zeta))*(1/(1-sim.eco.b))*(v**(sim.eco.b/(1-sim.eco.b)))+(sim.eco.firms.z**(-sim.eco.zeta))*sim.eco.lamb_a[1]
    
    nuage=[non_linear_eq_b_q_finites_2(j) for j in x_s]
    nuage_abs=[abs(i) for i in nuage]
    
    p_eq.append(fsolve(func=non_linear_eq_b_q_finites_2, x0=initial_guess_v, fprime=non_linear_eq_b_q_finites_2_prime)**(1/(1-sim.eco.coefficient)))
    
    #p_eq_m=minimize(fun=non_linear_eq_b_q_finites_2, x0=initial_guess_v, method="COBYLA")
    

    #print(p_eq_r)
    
    #plt.axvline(x=p_eq*10,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
    plt.axvline(x=nuage_abs.index(min(nuage_abs)),linewidth=1.3, alpha=1, color="yellow", label="p=p_eq")
    plt.axhline(y=0,linewidth=1.3, alpha=1, color="black", label="p=p_eq")
    plt.plot(nuage[:1000])
    #p_eq.append((fsolve(non_linear_eq_b_q_finites_2,initial_guess_v))**(1/(sim.eco.coefficient-1)))

