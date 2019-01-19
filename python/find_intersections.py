#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../python')
import post_process_hyperparameters
import plot_info
import print_table
import intersections


# # Intesections where speedup => 2 and prediction <0.05

# In[2]:


data_source = 'QMC_from_data'
convergence_rate = 0.75 # measured emperically
filenames = {'Drag' : '../data/qmc_airfoils_network_sizes_all_Drag.json',
             'Lift' : '../data/qmc_airfoils_network_sizes_all_Lift.json',
             'Q1' : '../data/qmc_sodshocktube_network_size_Q1.json',
             'Q2' : '../data/qmc_sodshocktube_network_size_Q2.json',
             'Q3':'../data/qmc_sodshocktube_network_size_Q3.json'
            }
intersections.find_intersections_acceptable(filenames, data_source, convergence_rate,
                                           min_speedup=2, max_prediction=0.05,
                                           print_filename='acceptable.json',
                                           table_filename='acceptable')


# In[ ]:
