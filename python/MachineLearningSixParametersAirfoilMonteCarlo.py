#!/usr/bin/env python
# coding: utf-8

# # Airfoil experiments
# All data is available in the repository

# In[1]:


import sys
sys.path.append('../python')

import matplotlib

import matplotlib.pyplot as plt

from machine_learning import *
import os
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes
from train_single_network import train_single_network

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# # Setup

# In[2]:



def get_airfoils_mc_data(highres=False):
    mc_points_preprocessed = np.loadtxt('../mc6.txt')
    if highres:
        forces = np.loadtxt('../force_L2_mc_scaled.dat')
    else:
        forces = np.loadtxt('../force_6_params_mc.dat')
    mc_points = []
    for f in forces[:,0]:
        for n in range(mc_points_preprocessed.shape[0]):
            if mc_points_preprocessed[n,0] == f:
                mc_points.append(mc_points_preprocessed[n,1:])
    mc_points = np.array(mc_points)

    force_names=['Lift', 'Drag']

    data_per_func = {}

    for n, force_name in enumerate(force_names):
        data_per_func[force_name] = forces[:,n+1]

    return mc_points, data_per_func

def get_airfoils_mc_network():
    airfoils_network = [12, 12, 10, 12, 10, 12, 10, 10, 12,1]

def get_airfoils_mc_data_highres_with_qmc():
    mc_points, mc_data_per_func = get_airfoils_mc_data(True)


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

    return mc_points, mc_data_per_func, qmc_points, data_per_func



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute airfoil case (with MC points)')


    parser.add_argument('--functional_name',
                        default=None,
                        help='The functional to use options: (lift or drag)')

    args = parser.parse_args()
    functional_name = args.functional_name

    parameters, data_per_func, validation_parameters, validation_values = get_airfoils_mc_data_highres_with_qmc()
    for n, force_name in enumerate(data_per_func):
        if functional_name is  None or (force_name.lower() == functional_name.lower()):
            display(HTML("<h1>%s</h1>"% force_name))
            try_best_network_sizes(parameters=parameters,
                                   samples=data_per_func[force_name],
                                   base_title='Airfoils MC %s' % force_name)


    for n, force_name in enumerate(data_per_func):
        if functional_name is  None or (force_name.lower() == functional_name.lower()):
            display(HTML("<h1>%s</h1>"% force_name))
            train_single_network(parameters=parameters,
                                 samples=data_per_func[force_name],
                                 base_title='Airfoils MC %s' % force_name,
                                 network = get_airfoils_mc_network(),
                                 large_integration_points = None,
                                 sampling_method='MC',
                                 monte_carlo_values = validation_values,
                                 monte_carlo_parameters  = validation_parameters
            )


    # In[ ]:
