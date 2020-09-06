#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
#sys.path.append('/mnt/research-live/user/cboissel/network-economy/src/')
sys.path.append('/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/network-economy/src')
import numpy as np
import random
import re

import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

from dynamics import Dynamics as dyn
from economy import Economy as eco

import firms 
import household


#%%

# SIMULATION

# %% CREATION ECONOMIE

def Variables_Simulation(alpha,alpha_p,beta,beta_p,w,q,b):
    sim_args={}
    
    # Variables statiques Firms
    sim_args["firms_args"]={
            "z":np.random.uniform(5,5,1),
            "sigma":np.random.uniform(0,1,1),
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
    p_init=5
    g_init=np.array([3])
    
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

# %% %% SIMULATION

def Simulation(**sim_args):
# Création objet classe Economy
    economie=eco(**sim_args["econ_args"])
    
    # Initialisations demandées à Economy
    economie.init_house(**sim_args["house_args"])
    economie.init_firms(**sim_args["firms_args"])
    
    economie.set_quantities()
    
    
    # Création de l'objet dynamique
    sim = dyn(t_max=500,e=economie)
    
    # Dynamique
    sim.discrete_dynamics(**sim_args["dyn_args"])
    
    return sim


#%%

# EQUILIBRIUM


#%%  COMPUTE EQUILIBRIUM
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

#%% CLASSIFY EQUILIBRIUM

def Classify_p_inf(sim,p_eq_0):
    diff=sim.prices[-101:-1]-sim.prices[-102:-2]
    if np.mean(diff)==0:
        if sim.prices[-10]==p_eq_0:
            p_inf="conv_eq"
        else:
            p_inf="conv_infl"
    
    elif np.mean(np.log(diff))>=10:
        p_inf="div_exp"
        
    else:
        p_inf="div"
    
    return p_inf
    
    
#%% PLOT EQUILIBRIUM ON PRICES 

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

#%% PLOT EQUILIBRIUM ON PRODUCTION 
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

#%% SIMULATIONS 

values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
q=0
b=1
directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q) 
#os.mkdir(directoire)                
behaviour={}
for alpha in values:
    for alpha_p in values:
        for beta in values:
            for beta_p in values:
                for w in values:
                    scenario="alpha="+str(alpha)+"_alpha_p="+str(alpha_p)+"_beta="+str(beta)+"_beta_p="+str(beta_p)+"_w="+str(w)         
                    sim_args=Variables_Simulation(alpha,alpha_p,beta,beta_p,w,q,b)
                    sim=Simulation(**sim_args)
                    p_eq_0,g_eq_0=Compute_Equilibrium(sim)
                    #Plot_PricesEq(sim, p_eq_0)
                    #Plot_ProductionEq(sim, p_eq_0)
                    behaviour[scenario]=Classify_p_inf(sim,p_eq_0)


#%% tentative de création d'un diagramme de stabilité 

def Plot_StabilityDiagramm(data_diagramme_x,data_diagramme_y,data_diagramme_be,beta,beta_p,w,nb_be):
            title="Stability Diagram. Types of behaviour:"+str(nb_be)+". \n beta="+str(beta)+"_"+"beta_p="+str(beta_p)+"_"+"w="+str(w) 
            fig, ax = plt.subplots()
            ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be)  
            ax.set_title(title)
            ax.set_xlabel("alpha")
            ax.set_ylabel("alpha_p")
            im=ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be) 
            fig.colorbar(im,ticks=[0,1,2,3])   
            fig.savefig(directoire+"/"+title+".png")
            
                    
for beta in values:
    for beta_p in values:
        for w in values:
            data_diagramme_x=[]
            data_diagramme_y=[]       
            data_diagramme_be=[]            
            for key in behaviour:
                if "beta="+str(beta) in key and "beta_p="+str(beta_p) in key and "w="+str(w) in key:
                    alpha=float(re.findall(r'alpha=(\d+\.\d+)_',key)[0])
                    alpha_p=float(re.findall(r'alpha_p=(\d+\.\d+)_',key)[0])
                    data_diagramme_x.append(alpha)
                    data_diagramme_y.append(alpha_p)
                    if behaviour[key]=="div":
                        data_diagramme_be.append(0) 
                    elif behaviour[key]=="div_exp":
                        data_diagramme_be.append(1) 
                    elif behaviour[key]=="conv_eq":
                        data_diagramme_be.append(2)
                    elif behaviour[key]=="conv_infl":
                        data_diagramme_be.append(3)
            nb_be=len(set(data_diagramme_be))
            Plot_StabilityDiagramm(data_diagramme_x,data_diagramme_y,data_diagramme_be,beta,beta_p,w,nb_be)


#%% GIF
from os import listdir
from os.path import isfile, join
file=directoire+"/"
filenames = [f for f in listdir(file) if isfile(join(file, f))]
filenames.remove('.DS_Store')
filenames.sort()            
import imageio
images = []

for filename in filenames:
    images.append(imageio.imread(file+filename))
imageio.mimsave(file+'essai.gif', images, duration=1)
