#!/usr/bin/env python
# coding: utf-8

# # Airfoil experiments
# All data is available in the repository

# In[1]:


import sys
sys.path.append('../python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from machine_learning import *
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes
from train_single_network import train_single_network

import MachineLearningSixParametersAirfoilMonteCarlo
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_airfoil_data():

    # # Setup

    # In[2]:


    qmc_points = np.loadtxt('../sobol_6_8000.txt')
    qmc_points = qmc_points[1:].reshape((8000,6))

    large_qmc_points = np.loadtxt('../sobol_6_131072.txt')
    all_points = qmc_points.copy()
    forces = np.array(np.loadtxt('../force_6_params.dat'))


    N = min(qmc_points.shape[0], forces.shape[0])
    qmc_points = qmc_points[:N,:]
    forces  = forces[:N,:]

    data_per_func = {}



    force_names = ['Lift', 'Drag']

    for n, force_name in enumerate(force_names):
        data_per_func[force_name] = forces[:, n+1]

    mc_params, mc_values = MachineLearningSixParametersAirfoilMonteCarlo.get_airfoils_mc_data(True)
    return qmc_points, data_per_func, mc_params, mc_values


def get_airfoils_network():
    return [12, 12, 10, 12, 10, 12, 10, 10, 12,1]

if __name__ == '__main__':
    airfoils_network = get_airfoils_network()


    parameters, data_per_func,_,_ = get_airfoil_data()


    for force_name in data_per_func.keys():
        try_best_network_sizes(parameters=parameters,
                               samples=data_per_func[force_name],
                               base_title='Airfoils %s' % force_name)




    for force_name in data_per_func.keys():
        train_single_network(parameters=parameters,
                             samples=data_per_func[force_name],
                             base_title='Airfoils %s' % force_name,
                             network = airfoils_network)
