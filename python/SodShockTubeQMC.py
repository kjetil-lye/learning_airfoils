#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from machine_learning import *
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes
from train_single_network import train_single_network
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import SodShockTube
# In[2]:
def get_sod_data_qmc():

    data = []
    with open('../data/SOD_QMC_DATA.dat', 'r') as inputfile:
        for l in inputfile:
            data.append([float(x) for x in l.split(',')])
    data = np.array(data)

    parameters = data[:,:6]
    samples = data[:,6:]

    func_names = ['Q1', 'Q2', 'Q3']

    data_by_func = {}

    mc_parameters, mc_data = SodShockTube.get_sod_data()
    for n, func_name in enumerate(func_names):
        data_by_func[func_name] =  samples[:,n]

    return parameters, data_by_func, mc_parameters, mc_data

def get_network():
    return [10, 10, 10, 10, 10,1]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute sodshocktube case (with QMC points)')


    parser.add_argument('--functional_name',
                        default=None,
                        help='The functional to use options: (q1, q2 or q3)')

    args = parser.parse_args()
    functional_name = args.functional_name


    network = get_network()

    parameters, data_by_func,_,_ = get_sod_data_qmc()

    for func_name in data_by_func.keys():
        if functional_name is  None or (func_name.lower() == functional_name.lower()):
            try_best_network_sizes(parameters=parameters,
                                   samples=data_by_func[func_name],
                                   base_title='Sod Shock QMC %s' % func_name)


    for func_name in data_by_func.keys():
        if functional_name is  None or (func_name.lower() == functional_name.lower()):
            train_single_network(parameters=parameters,
                                 samples=data_by_func[func_name],
                                 base_title='Sod Shock QMC %s' % func_name,
                                 network = network)


    # In[ ]:
