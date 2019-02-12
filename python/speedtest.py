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
import machine_learning
from train_single_network import compute_for_all_in_json
from post_process_hyperparameters import LatexWithAllPlots
from notebook_network_size import find_best_network_size_notebook, try_best_network_sizes_in_json
import argparse
import os
from data_sources import data_sources
import numpy as np
import time

parser = argparse.ArgumentParser(description='Does a speed test for the given network and functional.')
parser.add_argument('--data_source', type=str, default=None, action='append',
                    help='The datasource to choose from, possibilities: {}'.format("".join(["\t- {}\n".format(k) for k in data_sources.keys()])))

parser.add_argument('--json_file', type=str, default='../data/best_networks.json',
                    help='The JSON file to use to get the best networks from')

parser.add_argument('--functional_name', action='append',
                    default=None,
                    help='The functional to use')

parser.add_argument('--eval_tries', default=1024, type=int, help='Number of evaluations to do')

parser.add_argument('--output_table', default='runtimes', help='Output filename')

args = parser.parse_args()
json_file = args.json_file

if args.data_source is None:
    data_source_names = data_sources.keys()
else:
    data_source_names = args.data_source


for data_source_name in data_source_names:
    parameters, data_per_func, monte_carlo_parameters, monte_carlo_values = data_sources[data_source_name][0]()
    network = data_sources[data_source_name][1]()
    if args.functional_name is None:
        functional_names = data_per_func.keys()
    else:
        functional_names = args.functional_name
    for func_name in functional_names:
        table = print_table.TableBuilder()
        table.set_header(["step", "min runtime (s)", "max runtime (s)", "avg runtime (s)"])
        prefixes, networks = compute_for_all_in_json(json_file, parameters=parameters,
                               samples=data_per_func[func_name],
                               network=network,
                               base_title='speedtest_{} {}'.format(data_source_name, func_name),
                               monte_carlo_values = monte_carlo_values[func_name],
                               monte_carlo_parameters = monte_carlo_parameters)


        runtimes = []

        for prefix in prefixes:
            runtimes_new = np.load('results/' + prefix + "runtimes.npy")
            runtimes.append(runtimes_new)

        table.add_row(["training",
                       np.min(runtimes), np.max(runtimes), np.mean(runtimes)])




        eval_times = []

        eval_tries = args.eval_tries

        for model in networks:
            print(prefix)
            for t in range(eval_tries):
                sys.stdout.write("Try: {}\r".format(t))
                sys.stdout.flush()

                start = time.time()
                for q in range(50):
                    _ = model.predict(parameters)
                end = time.time()

                eval_times.append((end-start)/50)

            print()

        table.add_row(["evaluation (on {} parameters)".format(parameters.shape[0]),
                       np.min(eval_times), np.max(eval_times), np.mean(eval_times)])

        table.set_title("Runtimes for " + data_source_name + " " + func_name + " over network parameters in in {}".format(args.json_file))
        table.print_table(data_source_name + " " + func_name + args.output_table)
