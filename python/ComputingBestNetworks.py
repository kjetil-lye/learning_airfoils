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
import SodShockTube
import SodShockTubeQMC
import MachineLearningSixParametersAirfoil
import GaussianRandomVariable
from train_single_network import compute_for_all_in_json
from post_process_hyperparameters import LatexWithAllPlots
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes_in_json
import argparse

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

json_file = '../data/best_networks.json'


parser = argparse.ArgumentParser(description='Compute all test cases with the best networks')
parser.add_argument('--data_source', type=str, default=None,
                    help='The datasource to choose from, possibilities: {}'.format("".join(["\t- {}\n".format(k) for k in data_sources.keys()])))


parser.add_argument('--functional_name',
                    default=None,
                    help='The functional to use')

parser.add_argument('--load_weights', action='store_true',
                    help='Load the weights from file (assumes the corresponding \n\tresults/<prefix>model.json and\n\tresults/<prefix>model.h5\n exists)')

parser.add_argument('--try_network_sizes', action='store_true',
                    help='Do the analysis of the network sizes')
args = parser.parse_args()


if args.load_weights:
    print("Loading the weights from file")

latex_out = 'computing_best_networks.tex'
latex = LatexWithAllPlots()
plot_info.savePlot.callback = latex
print_table.print_comparison_table.callback = lambda x, title: latex.add_table(x, title)

if args.data_source is None:
    data_source_names = data_sources.keys()
else:
    data_source_names = [args.data_source]

for data_source_name in data_source_names:
    parameters, data_per_func, monte_carlo_parameters, monte_carlo_values = data_sources[data_source_name][0]()
    network = data_sources[data_source_name][1]()
    if args.functional_name is None:
        functional_names = data_per_func.keys()
    else:
        functional_names = [args.functional_name]
    for func_name in functional_names:

        if args.try_network_sizes:
            try_best_network_sizes_in_json(json_file, parameters=parameters,
                                   samples=data_per_func[func_name],
                                   base_title='{} {}'.format(data_source_name, func_name))
        else:
            compute_for_all_in_json(json_file, parameters=parameters,
                                   samples=data_per_func[func_name],
                                    network=network,
                                   base_title='{} {}'.format(data_source_name, func_name),
                                   monte_carlo_values = monte_carlo_values[func_name],
                                   monte_carlo_parameters = monte_carlo_parameters,
                                   load_network_weights = args.load_weights)
            try:
                compute_for_all_in_json(json_file, parameters=monte_carlo_parameters,
                                   samples=monte_carlo_values[func_name],
                                    network=network,
                                   base_title='MC {} {}'.format(data_source_name, func_name),
                                   monte_carlo_values = monte_carlo_values[func_name],
                                   monte_carlo_parameters = monte_carlo_parameters,
                                   load_network_weights = args.load_weights)
            except:
                pass

with open(latex_out, 'w') as f:
    f.write(latex.get_latex())


# In[ ]:
