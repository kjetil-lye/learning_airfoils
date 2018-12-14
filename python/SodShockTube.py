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


# In[2]:


data = []
with open('../SOD_MC_DATA.dat', 'r') as inputfile:
    for l in inputfile:
        data.append([float(x) for x in l.split(',')])
data = np.array(data)

parameters = data[:,:6]
samples = data[:,6:]


# In[3]:


epochs = 500000
network = [10, 10, 10, 10, 10,1]


# # Network sizes
# 

# In[4]:


func_names=['Q1', 'Q2', 'Q3']

for n, func_name in enumerate(func_names):
    try_best_network_sizes(parameters=parameters, 
                           samples=samples[:,n], 
                           base_title='Sod Shock MC %s' % func_name,
                          epochs=epochs)


# # Single network

# In[ ]:


func_names=['Q1', 'Q2', 'Q3']


for n, func_name in enumerate(func_names):
    train_single_network(parameters=parameters, 
                         samples=samples[:,n], 
                         base_title='Sod Shock MC %s' % func_name,
                         network = network,
                         epochs=epochs, 
                         large_integration_points = None,
                         sampling_method='MC')


# In[ ]:




