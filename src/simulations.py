#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
    Functions needed to run networks simulations in an aggregate manner:
        - stocks simulation variables 
        - runs simulation
        - computes and plots equilibria 
        - classifies type of equilibrium 
"""


# %%


import os,sys
sys.path.append('/mnt/research-live/user/cboissel/network-economy/src/')
#sys.path.append('/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/network-economy/src')
import numpy as np
import random
import re
import pandas as pd

import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

from dynamics import Dynamics as dyn
from economy import Economy as eco

import firms 
import household


# %%


### Stocks simulation variables changing only the parameters you want to work on
def Variables_Simulation(alpha,alpha_p,beta,beta_p,w,q,b,pert):
    sim_args={}
    
    # Variables statiques Firms
    sim_args["firms_args"]={
            "z":np.random.uniform(5,5,1),
            "sigma":np.random.uniform(0.2,0.2,1),
            "alpha":alpha,
            "alpha_p":alpha_p,
            "beta":beta,
            "beta_p":beta_p,
            "w":w
            }
    
    # Variables statiques Household
    sim_args["house_args"] = {
            'labour':10,
            'theta':np.ones(1),
            'gamma':1,
            "phi":1
            }
    
    # Variables statiques Economiy
    sim_args["econ_args"] = {
            'n':1,
            'd':1,
            'netstring':'regular',
            'directed':True,
            'j0':np.array([5]),
            'j1':np.array([2]),
            'a0':np.ones(1)*0.5,
            'q':q,
            'b':b
            }
    
    # Variables statiques Dynamics
    #p_init=3
    #g_init=np.array([5])
    p_init=1.6666666666666667+pert
    g_init=np.array([2])+pert
    
    sim_args["dyn_args"]={
            'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
            'w0':1,
            'g0':g_init,
            's0':np.random.uniform(0,0,1),
            't1':g_init,
            'B0':random.randint(0,0)
            }
    
    return sim_args


# %%


### Initializes and runs simulation
def Simulation(**sim_args):
# Création objet classe Economy
    economie=eco(**sim_args["econ_args"])
    
    # Initialisations demandées à Economy
    economie.init_house(**sim_args["house_args"])
    economie.init_firms(**sim_args["firms_args"])
    
    economie.set_quantities()
    
    
    # Création de l'objet dynamique
    sim = dyn(t_max=700,e=economie)
    
    # Dynamique
    sim.discrete_dynamics(**sim_args["dyn_args"])
    
    return sim


# %%


### Computes and stocks equilibrium values
def Compute_Equilibrium(sim):
    sim.eco.compute_eq()
    print("P_EQ", sim.eco.p_eq)
    print("G_EQ",sim.eco.g_eq)
    
    p_eq_0=sim.eco.p_eq[0]
    g_eq_0=sim.eco.g_eq[0]

    
    #g_eq_1=sim.eco.g_eq[1]
    #p_eq_1=sim.eco.p_eq[1]
    #p_eq_dis_1=sim.eco.p_eq[1] + random.uniform(-10**(-5),10**(-5))
    #g_eq_dis_1=sim.eco.g_eq[1] + random.uniform(-10**(-5),10**(-5))
    
    
    if sim.eco.b!=1 and sim.eco.q>0:
        print("ATTENTION A LA PRECISION DE L'EQUILIBRE")
        print(sim.prices[-1][0])
        prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]
        print(prod[-1])

    return p_eq_0, g_eq_0


# %%


### Plots equilibrium values

def Plot_PricesEq(sim, p_eq_0):
    ### Prices
    fig, ax = plt.subplots()
    ax.set_title("Prices of the firm's production")
    ax.set_xlabel('Time')
    ax.set_ylabel('P1')
    
    ax.plot(sim.prices[1:-1])
    if p_eq_0 >= 0:
        plt.axhline(y=p_eq_0,linewidth=1.3, alpha=1, color="green", label="p=p_eq")
    #plt.axhline(y=sim.p_eq_1,linewidth=1.3, alpha=1, color="red", label="p=p_eq")
    #plt.xscale("linear")

        
    ax.set_yscale("log")
    
    #plt.grid(True)
    #file=scenario+"_prices.png"              

    
    #fig.savefig(file)
    
    
def Plot_ProductionEq(sim,g_eq_0):
    ### Production
    fig, ax = plt.subplots()
    ax.set_title("Production")
    ax.set_xlabel('Time')
    ax.set_ylabel("P")
    
    prod=[sim.Q_demand[i,1,1]+sim.Q_demand[i,1,0] for i in range(len(sim.Q_demand))]
    
    ax.plot(prod[1:-1])
    
    if g_eq_0 >=0:
        plt.axhline(y=g_eq_0*sim.eco.firms.z,linewidth=1.3, alpha=1, color="green", label="prod=prod_eq")
    #plt.axhline(y=g_eq_1*sim.eco.firms.z,linewidth=1.3, alpha=1, color="red", label="prod=prod_eq")

    #plt.xscale("linear")s
    ax.set_yscale("log")
    #ax.set_ylim(0,float(max(prod))+100)
    #plt.grid(True)
    #file=scenario+"_prods.png" 
    #fig.savefig(file)


# %%


### Classify Equilibrium

# Typology
def Classify_p_inf(sim,p_eq_0, threshold=1e-6):
    std_diff=np.std(sim.prices[-101:-1]-sim.prices[-102:-2])
    if std_diff<=threshold:
        if np.abs(sim.prices[-10]*(sim.Q_real[-10][1,0])-sim.Q_real[-10][0,1])<=threshold:
            
            p_inf="conv_eq"
            
        else:
            print(sim.prices[-10]*(sim.Q_real[-10][1,0])-sim.Q_real[-10][0,1])
            p_inf="conv_infl"
            
    
    elif np.mean(np.log(std_diff))>=10:
        p_inf="div_exp"
        
    else:
        p_inf="div"
    
    return p_inf

# Exponent of the enxponential 
def Compute_ExpExponent(sim,t_max=500):

    return np.diff(np.array([float(i) for i in np.log(sim.prices[-101:-1])]),n=1)/np.diff(np.array(range(t_max-100,t_max)))[-1]

    
    


# %%


### Plot Stability Diagrams (colormap & scatterplot)

def Plot_StabilityDiagrammBe(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w,nb_be):
    title="Stability Diagram. Types of behaviour:"+str(nb_be)+". \n alpha="+str(alpha)+"_"+"alpha_p="+str(alpha_p)+"_"+"w="+str(w) 
    fig, ax = plt.subplots()
    ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be)  
    ax.set_title(title)
    ax.set_xlabel("beta_p")
    ax.set_ylabel("beta")
    ax.set_xlim(-0.05,1.05)
    im=ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be) 
    fig.colorbar(im,ax=ax)   # pour classification 
    fig.savefig(directoire+"/"+title+".png")

def Plot_StabilityDiagrammExp(data_diagramme_x,data_diagramme_y,data_diagramme_be,title,nb_be, values=values):
    coordonnees={}
    for i in range(len(values)):
        coordonnees[values[i]]=i
        
    data_slope=np.zeros((len(values),len(values)))
    for i in range(len(values)**2):
        data_slope[coordonnees[data_diagramme_y[i]],coordonnees[data_diagramme_x[i]]]=data_diagramme_be[i]

    fig, ax = plt.subplots()
    y, x = np.mgrid[0:11,0:11]/10
    im=ax.pcolor(x,y,data_slope, cmap='RdBu_r', norm=colors.SymLogNorm(linthresh=1e-9, linscale=1))
    cbar=fig.colorbar(im,ax=ax)
    cbar.ax.set_title("k")
    ax.set_title(title)
    ax.set_xlabel("alpha_p")
    ax.set_ylabel("beta_p")   
    fig.show()
    fig.savefig(directoire+"/"+title+".png")

            


# %%


### Creates a GIF with Stability Diagrams

def GIF_StabilityDiagram(i):
    values_ext=values*3
    alpha=values_ext[int(i/9)]
    alpha_p=values_ext[int(i/3)]
    w=values_ext[i]
    data_diagramme_x=[]
    data_diagramme_y=[]       
    data_diagramme_be=[]            
    for key in behaviour:
        if "alpha="+str(alpha)+"_" in key and "alpha_p="+str(alpha_p)+"_" in key and re.search("w="+str(w)+"$", key):
            beta_p=float(re.findall(r'beta_p=(\d+\.?\d*)_',key)[0])
            beta=float(re.findall(r'beta=(\d+\.?\d*)_',key)[0])
            data_diagramme_x.append(beta_p)
            data_diagramme_y.append(beta)
            data_diagramme_be.append(behaviour[key]) 
    nb_be=len(set(data_diagramme_be))
    Plot_StabilityDiagrammExp(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w)
    Plot_StabilityDiagrammBe(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w,nb_be)

