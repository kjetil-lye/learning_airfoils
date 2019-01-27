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

def get_airfoils_mc_data():
    mc_points_preprocessed = np.loadtxt('../mc6.txt')
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




if __name__ == '__main__':
    parameters, data_per_func = get_airfoils_mc_data()
    for n, force_name in enumerate(data_per_func):
        display(HTML("<h1>%s</h1>"% force_name))
        try_best_network_sizes(parameters=parameters,
                               samples=data_per_func[force_name],
                               base_title='Airfoils MC %s' % force_name)


    for n, force_name in enumerate(data_per_func):
        display(HTML("<h1>%s</h1>"% force_name))
        train_single_network(parameters=parameters,
                             samples=data_per_func[force_name],
                             base_title='Airfoils MC %s' % force_name,
                             network = get_airfoils_mc_network(),
                             large_integration_points = None,
                             sampling_method='MC')


    # In[ ]:
