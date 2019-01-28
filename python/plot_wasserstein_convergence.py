#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
sys.path.append('../python')
import post_process_hyperparameters
import plot_info
import print_table
import SodShockTube
import SodShockTubeQMC
import MachineLearningSixParametersAirfoil
import GaussianRandomVariable
from train_single_network import compute_for_all_in_json
from post_process_hyperparameters import LatexWithAllPlots
import scipy.stats
import matplotlib.pyplot as plt

# In[2]:


data_sources = {
    'Airfoils' : [
        MachineLearningSixParametersAirfoil.get_airfoil_data,
        MachineLearningSixParametersAirfoil.get_airfoils_network
    ],

    'SodShockTubeQMC' : [
        SodShockTubeQMC.get_sod_data_qmc,
        SodShockTubeQMC.get_network
    ],

    'Sine' : [
        GaussianRandomVariable.get_sine_data,
        GaussianRandomVariable.get_sine_network
    ]
}


for data_source_name in data_sources.keys():
    parameters, data_per_func, monte_carlo_parameters, monte_carlo_values = data_sources[data_source_name][0]()

    for func_name in data_per_func.keys():
        Ms = 2**np.arange(3, int(np.log2(parameters.shape[0])))
        errors = []
        for M in Ms:
            reference_size = 2**(int(np.log2(parameters.shape[0])))
            data = np.array(data_per_func[func_name])

            data_M_samples_upscaled = np.repeat(data[:M], reference_size/M, 0)

            errors.append(scipy.stats.wasserstein_distance(data[:reference_size], data_M_samples_upscaled))

        plt.loglog(Ms, errors, '-o', label='$W_1(\\mu^{M}, \\mu^{\\mathrm{Ref}})$')

        plt.xlabel('Number  of samples $M$')
        plt.ylabel("Wasserstein distance $W_1$")
        plt.title("Wasserstein convergence of {func_name} for {data_source_name}".format(func_name=func_name, data_source_name=data_source_name))

        poly = np.polyfit(np.log(Ms), np.log(errors), 1)
        plt.loglog(Ms, np.exp(poly[1])*Ms**poly[0], '--', label='$\\mathcal{O}(M^{%.2f})$' % poly[0])
        plt.legend()
        plot_info.showAndSave("wasserstein_convergence_{data_source_name}_{func_name}".format(func_name=func_name, data_source_name=data_source_name))

# In[ ]:
