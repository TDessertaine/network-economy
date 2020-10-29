#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os,sys
sys.path.append('/mnt/research-live/user/cboissel/network-economy/src/')
#sys.path.append('/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/network-economy/src')
import numpy as np
import random
import re
import pandas as pd
from os import listdir
from os.path import isfile, join 


import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

from dynamics import Dynamics as dyn
from economy import Economy as eco
from simulations import *

import firms 
import household


# %% CREATION ECONOMIE
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
    p_init=3
    g_init=np.array([5])
    
    sim_args["dyn_args"]={
            'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
            'w0':1,
            'g0':g_init,
            's0':np.random.uniform(0,0,1),
            't1':g_init,
            'B0':random.randint(0,0)
            }
    
    return sim_args

# Variables de suivi des perturbations pour un dossier donné (last: 20_08_2020)
state="nd"
compteur=5

# %%%% SIMULATION

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

# EQUILIBRIUM


# %% COMPUTE EQUILIBRIUM
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

def Disturb_Equilibrium(p_eq_0, g_eq_0):
    p_eq_dis_0=p_eq_0 + random.uniform(-10**(-5),10**(-5))
    g_eq_dis_0=g_eq_0 + random.uniform(-10**(-5),10**(-5))
    
    return p_eq_dis_0, g_eq_dis_0

# %% Verification
pert=random.uniform(-10**(-5),10**(-5))
        
# %time sim_args=Variables_Simulation(alpha=0.75,alpha_p=0.75,beta=0.75,beta_p=0.5,w=0.5,q=0,b=1,pert=pert)
# %time sim=Simulation(**sim_args)
#p_eq_0,g_eq_0=Compute_Equilibrium(sim)
#Plot_PricesEq(sim, p_eq_0)
#Plot_ProductionEq(sim, p_eq_0)
# %time Compute_ExpExponent(sim)

# %% PLOT EQUILIBRIUM ON PRICES

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

# %% PLOT EQUILIBRIUM ON PRODUCTION
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

# %% SIMULATIONS
values=[i/10 for i in range(11)]
values_alpha=[0.75]
values_alpha_p=[0.1]
values_w=[0.5]
q=0
b=1
pert=random.uniform(-10**(-5),10**(-5))
#directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q) 
#os.mkdir(directoire)   
directoire="/mnt/research-live/user/cboissel/network-economy/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q) 
#os.mkdir(directoire)                
behaviour={}
eq_infl={}
for alpha in values:
    for alpha_p in values:
        for w in values:
            for beta in values:
                for beta_p in values:
                    scenario="alpha="+str(alpha)+"_alpha_p="+str(alpha_p)+"_beta="+str(beta)+"_beta_p="+str(beta_p)+"_w="+str(w)         
                    print(scenario)
                    sim_args=Variables_Simulation(alpha,alpha_p,beta,beta_p,w,q,b,pert)
                    sim=Simulation(**sim_args)
                    p_eq_0,g_eq_0=Compute_Equilibrium(sim)
                    #Plot_PricesEq(sim, p_eq_0)
                    #Plot_ProductionEq(sim, p_eq_0)
                    if Classify_p_inf(sim,p_eq_0, threshold=1e-3)=="conv_infl":
                        eq_infl[scenario]=float(Compute_ExpExponent(sim)[-1])
                        print("EQ_INF")
                    behaviour[scenario]=float(Compute_ExpExponent(sim)[-1])
pd.DataFrame.from_dict(behaviour, orient="index").to_csv(directoire+'/11ValuesExponentsPerturbedEq.csv', header=False, index=range(len(behaviour)))


# %%
print(sim.eco.compute_eq())

# %%
values_alpha=[i/20 for i in range(4,21)]
print(values_alpha)

# %% SAVE DATA
pd.DataFrame.from_dict(behaviour, orient="index").to_csv(directoire+'/2_1ValuesExponentsPerturbedEq.csv', header=False, index=range(len(behaviour)))

# %%
import pandas as pd
directoire="/mnt/research-live/user/cboissel/network-economy/2020_09_04_Scenarii_b="+str(1)+"_q="+str(0)
data=pd.read_csv(directoire+'/11ValuesExponentsPerturbedEq.csv')

behaviour={'alpha=0_alpha_p=0_beta=0_beta_p=0_w=0':0.0}
for i in range(len(data)):
    behaviour[data.iat[i,0]]=data.iat[i,1]

# %%
### REPRESENTATIONS GRAPHIQUES AVEC CLASSIFY

# %% tentative de création d'un diagramme de stabilité
values=[i/10 for i in range(11)]
#values=[i/100 for i in range(101)]
q=0
b=0.9
#directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_09_04_Scenarii_b=1_q=0"
directoire="/mnt/research-live/user/cboissel/network-economy/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q)+"/AlphapBeta_p"
os.mkdir(directoire) 
def Plot_StabilityDiagrammBe(data_diagramme_x,data_diagramme_y,data_diagramme_be,title,nb_be):
    fig, ax = plt.subplots()
    ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be)  
    ax.set_title(title)
    ax.set_xlabel("beta_p")
    ax.set_ylabel("beta")
    ax.set_xlim(-0.05,1.05)
    im=ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be) 
    fig.colorbar(im,ax=ax)   # pour classification 
    #fig.savefig(directoire+"/"+title+".png")


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
    

for alpha in values:
    for beta in values:
        for w in values:
            data_diagramme_x=[]
            data_diagramme_y=[]       
            data_diagramme_be=[]            
            for key in behaviour:
                if "alpha="+str(alpha)+"_" in key and "beta="+str(beta)+"_" in key and re.search("w="+str(w)+"$", key):#
                    x_axis=float(re.findall(r'alpha_p=(\d+\.?\d*)_',key)[0])
                    y_axis=float(re.findall(r'beta_p=(\d+\.?\d*)_',key)[0])
                    data_diagramme_x.append(x_axis)
                    data_diagramme_y.append(y_axis)
                    data_diagramme_be.append(behaviour[key]) 
            nb_be=len(set(data_diagramme_be))
            title="Stability Diagram. Types of behaviour:"+str(nb_be)+". \n alpha="+str(alpha)+"_"+"beta="+str(beta)+"_"+"w="+str(w) 
            Plot_StabilityDiagrammExp(data_diagramme_x,data_diagramme_y,data_diagramme_be,title,nb_be,values)


# %%
imageio.mimsave(directoire+'/11ValeursBetaBeta_p.gif', images, duration=1)

# %% GIF
# GIF AVEC imageio
file=directoire+"/"
filenames = [f for f in listdir(file) if isfile(join(file, f))]
filenames.remove('.DS_Store')
filenames.remove('5ValuesBehaviourEquilibrium.csv')
filenames.remove('5ValuesBehaviourPerturbedEq.csv')
filenames.remove("5ValeursAlphaAlpha_p.gif")
filenames.remove("5ValeursAlphaBeta.gif")
filenames.remove('5ValeursAlpha_pBeta.gif')
filenames.remove('5ValeursAlphaBeta_p.gif')
filenames.remove('5ValeursAlphaW.gif')
filenames.remove('5ValeursAlpha_pW.gif')
filenames.remove('5ValeursAlpha_pBeta_p.gif')
#filenames.remove('5ValeursBetaBeta_p.gif')
filenames.remove('5ValeursBeta_pW.gif')
filenames.remove('5ValeursBetaW.gif')
filenames.sort()            

images = []

for filename in filenames:
    images.append(imageio.imread(file+filename))
imageio.mimsave(file+'5ValeursBetaBeta.gif', images, duration=1)
