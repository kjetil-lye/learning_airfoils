#!/usr/bin/env python
# coding: utf-8

# # Gaussian experiment
# All data is generated on the fly

# In[1]:


import sys
sys.path.append('../python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from machine_learning import *
import scipy.stats
import os
import sobol
import resource
import json
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes
from train_single_network import train_single_network
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def generate_sobol_points(M, dim):
    points = []
    for i in range(M):
        points.append(sobol.i4_sobol(dim,i)[0])
    return np.array(points)

def get_sine_data():
    dims = [6]
    dim = 6
    M = int(2**20)

    parameters =  generate_sobol_points(M, dim)


    functionals = {
               "Sine" : sine_functional,
               "Sine/d" : sine_functional_1,
               "Sine/d3" : sine_functional_3
        }



    data_per_func = {}


    for functional_name in functionals.keys():

        data_per_func["{}".format(functional_name)] = functionals[functional_name](parameters)

    return parameters, data_per_func


def get_sine_network():
    network_width = 12
    network_depth = 10

    gaussian_network =  [network_width for k in range(network_depth)]
    gaussian_network.append(1)

    return gaussian_network

def sine_functional(x):
    return np.sum(np.sin(4*np.pi*x), 1)

def sine_functional_1(x):
    return np.sum(np.sin(4*np.pi*x)/np.arange(1, dim+1), 1)

def sine_functional_3(x):
    return np.sum(np.sin(4*np.pi*x)/np.arange(1,dim+1)**3, 1)





# # Single network

# In[ ]:


if __name__ == '__main__':
    parameters, data_per_func = get_sine_data()
    for functional_name in data_per_func.keys():
        title = '%s %s' % (data_source_name, functional_name)
        display(HTML("<h1>%s</h1>" % title))
        train_single_network(parameters=parameters,
                           samples=data_per_func[functional_name],
                         base_title=title,
                         network = get_sine_network(),
                         large_integration_points = None,
                         sampling_method=sampling_method)

        title = '%s %s' % (data_source_name, functional_name)
        display(HTML("<h1>%s</h1>" % title))
        try_best_network_sizes(parameters=parameters,
                           samples=data_per_func[functional_name],
                           base_title=title)
    # In[ ]:





    # In[ ]:
