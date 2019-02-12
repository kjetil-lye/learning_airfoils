#!/usr/bin/env python
# coding: utf-8
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
import os

from data_sources import data_sources


parser = argparse.ArgumentParser(description='Compute all test cases with the best networks')
parser.add_argument('--data_source', type=str, default=None, action='append',
                    help='The datasource to choose from, possibilities: {}'.format("".join(["\t- {}\n".format(k) for k in data_sources.keys()])))

parser.add_argument('--json_file', type=str, default='../data/best_networks.json',
                    help='The JSON file to use to get the best networks from')

parser.add_argument('--functional_name', action='append',
                    default=None,
                    help='The functional to use')


parser.add_argument('--do_not_train_single_size_first', action='store_true',
                    help='By default, we always train a single small network first for each functional. Add this option to disable this.')

parser.add_argument('--load_weights', action='store_true',
                    help='Load the weights from file (assumes the corresponding \n\tresults/<prefix>model.json and\n\tresults/<prefix>model.h5\n exists)')

parser.add_argument('--try_network_sizes', action='store_true',
                    help='Do the analysis of the network sizes (instead of normal training).')

parser.add_argument('--train_monte_carlo', action='store_true',
                    help='By default we train with QMC points, enabling this option trains it with MC points (network sizes are always with QMC for now)')
args = parser.parse_args()


if args.load_weights:
    print("Loading the weights from file")

latex_out = 'computing_best_networks.tex'
latex = LatexWithAllPlots()
plot_info.savePlot.callback = latex
print_table.print_comparison_table.callback = lambda x, title: latex.add_table(x, title)
json_file = args.json_file

if args.data_source is None:
    data_source_names = data_sources.keys()
else:
    data_source_names = args.data_source


if not args.try_network_sizes and not args.do_not_train_single_size_first:
    for data_source_name in data_source_names:
        parameters, data_per_func, monte_carlo_parameters, monte_carlo_values = data_sources[data_source_name][0]()
        network = data_sources[data_source_name][1]()
        if args.functional_name is None:
            functional_names = data_per_func.keys()
        else:
            functional_names = args.functional_name
        for func_name in functional_names:

            os.environ['MACHINE_LEARNING_NUMBER_OF_WIDTHS'] = '1'
            os.environ['MACHINE_LEARNING_NUMBER_OF_DEPTHS'] = '1'
            try_best_network_sizes_in_json(json_file, parameters=parameters,
                                   samples=data_per_func[func_name],
                                   base_title='{} {}'.format(data_source_name, func_name))

for data_source_name in data_source_names:
    parameters, data_per_func, monte_carlo_parameters, monte_carlo_values = data_sources[data_source_name][0]()
    network = data_sources[data_source_name][1]()
    if args.functional_name is None:
        functional_names = data_per_func.keys()
    else:
        functional_names = args.functional_name
    for func_name in functional_names:

        if args.try_network_sizes:
            try_best_network_sizes_in_json(json_file, parameters=parameters,
                                   samples=data_per_func[func_name],
                                   base_title='{} {}'.format(data_source_name, func_name))
        else:
            if not args.train_monte_carlo:
                compute_for_all_in_json(json_file, parameters=parameters,
                                       samples=data_per_func[func_name],
                                        network=network,
                                       base_title='{} {}'.format(data_source_name, func_name),
                                       monte_carlo_values = monte_carlo_values[func_name],
                                       monte_carlo_parameters = monte_carlo_parameters,
                                       load_network_weights = args.load_weights)
            else:
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
