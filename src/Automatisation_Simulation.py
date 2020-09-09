#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
# #!pip install matplotlib
# !pip install scipy
import os,sys
sys.path.append('/mnt/research-live/user/cboissel/network-economy/src/')
#sys.path.append('/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/network-economy/src')
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


# %%

# SIMULATION

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

# %% CLASSIFY EQUILIBRIUM

def Classify_p_inf(sim,p_eq_0, threshold=1e-6):
    std_diff=np.std(sim.prices[-101:-1]-sim.prices[-102:-2])
    if std_diff<=threshold:
        if np.abs(sim.prices[-10]-p_eq_0)<=threshold:
            p_inf="conv_eq"
            
        else:
            p_inf="conv_infl"
            
    
    elif np.mean(np.log(std_diff))>=10:
        p_inf="div_exp"
        
    else:
        p_inf="div"
    
    return p_inf


def Compute_ExpExponent(sim,t_max=500):

    return np.diff(np.array([float(i) for i in np.log(sim.prices[-101:-1])]),n=1)/np.diff(np.array(range(t_max-100,t_max)))[-1]

    
    

# %% Verification
        
sim_args=Variables_Simulation(alpha=0.75,alpha_p=0.75,beta=0.75,beta_p=0.5,w=0.5,q=0,b=1,pert=pert)
sim=Simulation(**sim_args)
#p_eq_0,g_eq_0=Compute_Equilibrium(sim)
#Plot_PricesEq(sim, p_eq_0)
#Plot_ProductionEq(sim, p_eq_0)
Classify_p_inf(sim,p_eq_0)


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

values=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
q=0
b=1
pert=random.uniform(-10**(-5),10**(-5))
#directoire="/Users/boisselcamille/Documents/Stage_Econophysix/networks_code/OneFirmCase_Images_v1/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q) 
#os.mkdir(directoire)   
directoire="/mnt/research-live/user/cboissel/network-economy/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q) 
#os.mkdir(directoire)                
behaviour={}
for alpha in values:
    for alpha_p in values:
        for beta in values:
            for beta_p in values:
                for w in values:
                    scenario="alpha="+str(alpha)+"_alpha_p="+str(alpha_p)+"_beta="+str(beta)+"_beta_p="+str(beta_p)+"_w="+str(w)         
                    print(scenario)
                    sim_args=Variables_Simulation(alpha,alpha_p,beta,beta_p,w,q,b,pert)
                    sim=Simulation(**sim_args)
                    #p_eq_0,g_eq_0=Compute_Equilibrium(sim)
                    #Plot_PricesEq(sim, p_eq_0)
                    #Plot_ProductionEq(sim, p_eq_0)
                    #behaviour[scenario]=Classify_p_inf(sim,p_eq_0, threshold=1e-6)
                    behaviour[scenario]=float(Compute_ExpExponent(sim)[-1])


# %% SAVE DATA
import pandas as pd
pd.DataFrame.from_dict(behaviour, orient="index").to_csv(directoire+'/11ValuesExponentsPerturbedEq.csv', header=False, index=range(len(behaviour)))

# %%
                    
   ### REPRESENTATIONS GRAPHIQUES AVEC CLASSIFY

# %% tentative de création d'un diagramme de stabilité

def Plot_StabilityDiagrammBe(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w,nb_be):
    title="Stability Diagram. Types of behaviour:"+str(nb_be)+". \n alpha="+str(alpha)+"_"+"alpha_p="+str(alpha_p)+"_"+"w="+str(w) 
    fig, ax = plt.subplots()
    ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be)  
    ax.set_title(title)
    ax.set_xlabel("beta_p")
    ax.set_ylabel("beta")
    im=ax.scatter(data_diagramme_x,data_diagramme_y,c=data_diagramme_be) 
   # fig.colorbar(im,ticks=[0,1,2,3])   pour classification 
    fig.savefig(directoire+"/"+title+".png")

def Plot_StabilityDiagrammExp(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w,values=values):
    coordonnees={}
    for i in range(len(values)):
        coordonnees[values[i]]=i
        
    data_slope=np.zeros((len(values),len(values)))
    for i in range(len(values)):
        for j in range(len(values)):
            data_slope[coordonnees[data_diagramme_x[i+j]],coordonnees[data_diagramme_x[i+j]]]=data_diagramme_be[i+j]
    print(data_slope)
    title= "Stability Diagram. \n alpha="+str(alpha)+"_"+"alpha_p="+str(alpha_p)+"_"+"w="+str(w) 
    fig, ax = plt.subplots()
    im=ax.pcolor(data_slope) 
    fig.colorbar(im,ax=ax) 
    ax.set_title(title)
    ax.set_xlabel("beta_p")
    ax.set_ylabel("beta")   
    #fig.savefig(directoire+"/"+title+".png")

            
                    
for alpha in values:
    for alpha_p in values:
        for w in values:
            data_diagramme_x=[]
            data_diagramme_y=[]       
            data_diagramme_be=[]            
            for key in behaviour:
                if "alpha="+str(alpha) in key and "alpha_p="+str(alpha_p) in key and "w="+str(w) in key:
                    beta_p=float(re.findall(r'beta_p=(\d+\.\d+)_',key)[0])
                    beta=float(re.findall(r'beta=(\d+\.\d+)_',key)[0])
                    data_diagramme_x.append(beta_p)
                    data_diagramme_y.append(beta)
                    data_diagramme_be.append(behaviour[key]) 
            nb_be=len(set(data_diagramme_be))
            
            Plot_StabilityDiagrammExp(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w)
            Plot_StabilityDiagrammBe(data_diagramme_x,data_diagramme_y,data_diagramme_be,alpha,alpha_p,w,nb_be)


# %% GIF
from os import listdir
from os.path import isfile, join
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
import imageio
images = []

for filename in filenames:
    images.append(imageio.imread(file+filename))
imageio.mimsave(file+'5ValeursBetaBeta.gif', images, duration=1)

# %%

    ### STATS DES

# %% Type de convergence en fonction de la somme des paramètres

## Dictionnaire
sum_param={}
sum_param["div"]=[]
sum_param["div_exp"]=[]
sum_param["conv_eq"]=[]
sum_param["conv_infl"]=[]

for key in behaviour:
    somme=sum([float(i) for i in re.findall(r'=(\d+\.\d+)',key)])
    sum_param[behaviour[key]].append(somme)

# %%

## Stats des
import statistics as stats

for i in sum_param:
    print(i, "effectifs=",len(sum_param[i]), "moyenne=", stats.mean(sum_param[i]))
    print("quartiles=",np.quantile(np.array(sum_param[i]),q=[0.25,0.5,0.75]))
    fig1,ax1=plt.subplots()
    ax1.set_title(i+"_Distribution")
    ax1.hist(sum_param[i])
    fig2,ax2=plt.subplots()
    ax2.set_title(i+"_Distribution")
    prob=[j for j in range(len(sum_param[i]))]
    ax2.plot(sorted(sum_param[i]), prob)

# %% Type de convergence en fonction de chaque paramètre
    
## Initialisation
alpha=[]
beta=[]
alpha_p=[]
beta_p=[]
w=[]
sum_param_list=[]
be=[]
for key in behaviour:
    alpha.append(float(re.findall(r'alpha=(\d+\.\d+)_',key)[0]))
    beta.append(float(re.findall(r'beta=(\d+\.\d+)_',key)[0]))
    alpha_p.append(float(re.findall(r'alpha_p=(\d+\.\d+)_',key)[0]))
    beta_p.append(float(re.findall(r'beta_p=(\d+\.\d+)_',key)[0]))
    w.append(float(re.findall(r'w=(\d+\.\d+)',key)[0]))
    sum_param_list.append(sum([float(i) for i in re.findall(r'=(\d+\.\d+)',key)]))
    if behaviour[key]=="div":
        be.append(0) 
    elif behaviour[key]=="div_exp":
        be.append(1) 
    elif behaviour[key]=="conv_eq":
        be.append(2)
    elif behaviour[key]=="conv_infl":
        be.append(3)
    
    
    
# Alpha
fig1,ax1=plt.subplots(figsize=(6, 10))
ax1.set_title("Behaviour of economy: alpha and sum parameters")
ax1.set_xlabel("alpha")
ax1.set_ylabel("Sum of Parameters")
im1=ax1.scatter(alpha,sum_param_list,c=be) 
fig1.colorbar(im1,ticks=[0,1,2,3])   
fig1.savefig(directoire+"/Behaviour of economy: alpha and sum parameters.png")

# Alpha_p
fig1,ax1=plt.subplots(figsize=(6, 10))
ax1.set_title("Behaviour of economy: alpha_p and sum parameters")
ax1.set_xlabel("alpha_p")
ax1.set_ylabel("Sum of Parameters")
im1=ax1.scatter(alpha_p,sum_param_list,c=be) 
fig1.colorbar(im1,ticks=[0,1,2,3]) 
fig1.savefig(directoire+"/Behaviour of economy: alpha_p and sum parameters.png") 

# Beta
fig1,ax1=plt.subplots(figsize=(6, 10))
ax1.set_title("Behaviour of economy: beta and sum parameters")
ax1.set_xlabel("beta")
ax1.set_ylabel("Sum of Parameters")
im1=ax1.scatter(beta,sum_param_list,c=be) 
fig1.colorbar(im1,ticks=[0,1,2,3]) 
fig1.savefig(directoire+"/Behaviour of economy: beta and sum parameters.png") 

# Beta_p
fig1,ax1=plt.subplots(figsize=(6, 10))
ax1.set_title("Behaviour of economy: beta_p and sum parameters")
ax1.set_xlabel("beta_p")
ax1.set_ylabel("Sum of Parameters")
im1=ax1.scatter(beta_p,sum_param_list,c=be) 
fig1.colorbar(im1,ticks=[0,1,2,3])
fig1.savefig(directoire+"/Behaviour of economy: beta_p and sum parameters.png")  

# W
fig1,ax1=plt.subplots(figsize=(6, 10))
ax1.set_title("Behaviour of economy: w and sum parameters")
ax1.set_xlabel("w")
ax1.set_ylabel("Sum of Parameters")
im1=ax1.scatter(w,sum_param_list,c=be) 
fig1.colorbar(im1,ticks=[0,1,2,3]) 
fig1.savefig(directoire+"/Behaviour of economy: w and sum parameters.png")  
# %% Stats des

