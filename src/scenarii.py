#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
Interesting behaviours.
"""


# %%


### Oscillations

# Oscillations resserrées 
alpha=0.3
alpha_p=0.05
w=0.1

beta=0.6
beta_p=0.05

b=1.0
q=0.0
pert = -8.864042540728256e-07
sim_args = {}
# Variables statiques Firms
sim_args["firms_args"] = {
    "z":np.random.uniform(12, 12, 1),
    "sigma":np.random.uniform(0.2, 0.2, 1),
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
p_init = 0.5+pert
g_init = np.array([2])+pert
sim_args["dyn_args"] = {
    'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
    'w0':1,
    'g0':g_init,
    's0':np.random.uniform(0, 0, 1),
    't1':g_init,
    'B0':random.randint(0, 0)
    }



# Oscillations plus lâches 
alpha=0.3
alpha_p=0.05
w=0.1

beta=0.3
beta_p=0.05

b=1.0
q=0.0
pert = -8.864042540728256e-07
sim_args = {}
# Variables statiques Firms
sim_args["firms_args"] = {
    "z":np.random.uniform(12, 12, 1),
    "sigma":np.random.uniform(0.2, 0.2, 1),
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
p_init = 0.5+pert
g_init = np.array([2])+pert
sim_args["dyn_args"] = {
    'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
    'w0':1,
    'g0':g_init,
    's0':np.random.uniform(0, 0, 1),
    't1':g_init,
    'B0':random.randint(0, 0)
    }


# %%


#### Equilibre inflationnaire
alpha=0.3
alpha_p=0.05
w=0.3

beta=0.8
beta_p=0.05

b=1.0
q=0.0
pert=rd.uniform(-10**(-6),10**(-6))

p_init=0.75
g_init=np.array([1])
sim_args = {}
# Variables statiques Firms
sim_args["firms_args"] = {
    "z":np.random.uniform(12, 12, 1),
    "sigma":np.random.uniform(0.2, 0.2, 1),
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
sim_args["dyn_args"] = {
    'p0':np.array([p_init]),#np.random.uniform(1,2,econ_args['n']),
    'w0':1,
    'g0':g_init,
    's0':np.random.uniform(0, 0, 1),
    't1':g_init,
    'B0':random.randint(0, 0)
    }


# %%
#### Chaos ??
alpha=0.3
alpha_p=0.05
w=0.1

beta=0.1
beta_p=0.05

b=1
q=0
pert = 0
directoire="/mnt/research-live/user/cboissel/network-economy/2020_09_04_Scenarii_b="+str(b)+"_q="+str(q)+"/PhaseDiagrams" 
#os.mkdir(directoire)

scenario="alpha="+str(alpha)+"_alpha_p="+str(alpha_p)+"_beta="+str(beta)+"_beta_p="+str(beta_p)+"_w="+str(w)

values_p=np.logspace(-3,4,15)
values_g=[np.array([i]) for i in values_p]

p_init=values_p[14]
g_init=values_g[14]

