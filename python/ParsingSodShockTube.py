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


# In[ ]:


plot_info.showAndSave.prefix = 'sodshocktube'
plot_info.savePlot.saveTikz = False
print_table.print_comparison_table.silent = True
convergence_rate = 0.75 # measured emperically
filenames = {'Q1' : '../data/qmc_sodshocktube_network_size_Q1.json',
             'Q2' : '../data/qmc_sodshocktube_network_size_Q2.json',
             'Q3':'../data/qmc_sodshocktube_network_size_Q3.json'}
post_process_hyperparameters.plot_all(filenames, convergence_rate, 'sodshocktube.tex')


# In[ ]:
