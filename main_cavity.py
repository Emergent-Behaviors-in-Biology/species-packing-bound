# -*- coding: utf-8 -*-
"""
Created on Thu 03/31/2019

@author: Wenping Cui
"""
import time
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Eco_function.eco_lib import *
from Eco_function.eco_plot import *
from Eco_function.eco_func import *
from Eco_function.Model_cavity import *
from Eco_function.usertools import MakeMatrices
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
import argparse
parser = argparse.ArgumentParser(description='Process types and dynamics')
parser.add_argument('--B', default='null', choices=['block', 'null', 'circulant','identity'])
parser.add_argument('--C', default='gaussian', choices=['gaussian','binomial', 'uniform'])
parser.add_argument('--d', default='quadratic', choices=['linear','quadratic', 'crossfeeding'])
parser.add_argument('--s', default='CVXOPT', choices=['ODE', 'CVXOPT'])
parser.add_argument('--m', default='null', choices=['null','add', 'scale', 'power']) # 'add', 'power', 'null'
args = parser.parse_args()
dynamics=  args.d 
B_type = args.B   
C_type = args.C  
Simulation_type=args.s 
Metabolic_Tradeoff_type=args.m

start_time = time.time()
Pool_num=15
file_name='Community_'+C_type+'_'+B_type +'_'+dynamics+'_'+Simulation_type+'_'+Metabolic_Tradeoff_type+'_linear_sigc_1.csv'

parameters = {}
parameters['sample_size']=10;
parameters['S'] =100;
parameters['M']=100;

parameters['K']=10.0;
parameters['sigma_K']=1.0;

parameters['mu']=1.0;
parameters['sigma_c']=2.0; 

parameters['m']=1.;
parameters['sigma_m']=0.1;
parameters['loop_size']=50;


parameters['t0']=0;
parameters['t1']=500;
parameters['Nt']=1000;
def func_parallel(para):
	parameter = {}
	parameter['sample_size']=para[0];
	parameter['S'] =para[1];
	parameter['M']=para[2];

	parameter['K']=para[3];
	parameter['sigma_K']=para[4];

	parameter['mu']=para[5];
	parameter['sigma_c']=para[6]; 

	parameter['m']=para[7];
	parameter['sigma_m']=para[8];
	parameter['loop_size']=para[9];


	parameter['t0']=para[10];
	parameter['t1']=para[11];
	parameter['Nt']=para[12];

	epsilon=para[13]
	mu=para[14]
	D=para[15]
	parameter['w']=para[16]
	parameter['tau_inv']=parameter['w']

	Model=Cavity_simulation(parameter)
	Model.Bnormal=False
	Model.gamma_flag='S/M'
	if B_type=='identity': #'diag', 'null', 'circulant' and 'block'
		Model.B_type='identity'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='null':
		Model.B_type='null'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='circulant':
		Model.B_type='circulant'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='block':
		Model.B_type='block'
		Model.mu=mu
		Model.epsilon=epsilon
	if C_type=='binomial':
		Model.C_type='binomial'
		Model.p_c=epsilon
		Model.mu = mu
		Model.epsilon= epsilon
	elif C_type=='gamma':
		Model.C_type='gamma'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='gaussian':
		Model.C_type='gaussian'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='uniform':
		Model.C_type='uniform'
		Model.mu=mu
		Model.epsilon=epsilon
	if Metabolic_Tradeoff_type!='null':  # 'scale', 'add', 'power'
		Model.Metabolic_Tradeoff=True
		Model.Metabolic_Tradeoff_type=Metabolic_Tradeoff_type
		Model.mu=mu
		Model.epsilon=0.1
		Model.p_c=0.1
		Model.epsilon_Metabolic=epsilon # noise amplitude for soft metabolic trade-off
	if dynamics=='linear': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	elif dynamics=='quadratic': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	elif dynamics=='crossfeeding': #'quadratic' 'linear'
		Model.D =D
		Model.e=0.6
		Model.flag_crossfeeding=True
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	mean_var['dynamics']=dynamics
	mean_var['size']=parameter['S']
	mean_var['K']=para[3]
	mean_var['mu']=mu
	mean_var['M']=para[2];
	mean_var['S']=para[1]
	mean_var['epsilon']=Model.epsilon
	mean_var['sample_size']=parameter['sample_size']
	mean_var['epsilon_Metabolic']=Model.epsilon_Metabolic
	index = [0]
	para_df = pd.DataFrame(mean_var, index=index)


	filename='abundance'+'K_'+str(parameter['K'])+'Sigc_'+str(round(epsilon,3))+Simulation_type+Metabolic_Tradeoff_type+'.pkl'
	with open(filename, 'wb') as f:  
   		 pickle.dump((Model.R_org, Model.N_org, Model.packing, Model.lams ), f)
	return para_df

jobs=[];
for S in [100]:
	parameters['S'] =5*S;
	parameters['M'] =S
	parameters['sample_size']=int(100*1000/S);
	#for mu in np.append(0,np.logspace(-3.0, 2., num=10)):
	#for mu in [0, 0.6, 1.0, 3.0, 5.0, 8.0, 10.0]:
	mu=1.0
	w=1.
	for K in [10.]:  
		parameters['tau_inv']=w
		parameters['w']=w
		parameters['K']=K/parameters['tau_inv']
		parameters['sigma_K']=0.1/np.sqrt(parameters['tau_inv'])
		ranges=np.logspace(-6.0, -0.5, num=15)# 
		ranges=np.arange(0.0, 0.3, 0.01)   
		for epsilon in ranges: 
			jobs.append([parameters['sample_size'],parameters['S'],parameters['M'],parameters['K'],parameters['sigma_K'], parameters['mu'], parameters['sigma_c'],parameters['m'],parameters['sigma_m'],parameters['loop_size'],parameters['t0'],parameters['t1'],parameters['Nt']  ,epsilon, mu, 0,  parameters['w']])
pool = Pool(processes=Pool_num)
results = pool.map(func_parallel, jobs)
pool.close()
pool.join()
results_df = pd.concat(results)
with open(file_name, 'a') as f:
		results_df.to_csv(f, index=False,encoding='utf-8')




