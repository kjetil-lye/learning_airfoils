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


plot_info.showAndSave.prefix = 'airfoils_qmc'
plot_info.savePlot.saveTikz = False
print_table.print_comparison_table.silent = True
convergence_rate = 0.75 # measured emperically
filenames = {'Drag' : '../data/qmc_airfoils_network_sizes_all_Drag.json',
             'Lift' : '../data/qmc_airfoils_network_sizes_all_Lift.json'}
post_process_hyperparameters.plot_all(filenames, convergence_rate, 'airfoils.tex')


# In[ ]:
