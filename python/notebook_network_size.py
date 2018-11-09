import numpy as np
from mpi4py import MPI
import tensorflow as tf
from machine_learning import random_seed, seed_random_number, print_memory_usage
from plot_info import *
from print_table import *
import time


def find_best_network_size_notebook(*, network_information,
    output_information,
    train_size,
    run_function,
    number_of_depths,
    number_of_widths,
    base_title,
    only_selection):


    base_width=6
    base_depth=4
    widths = base_width*2**np.arange(0, number_of_widths)
    depths = base_depth*2**np.arange(0, number_of_depths)

    all_depths = base_depth * 2**np.arange(0, number_of_depths+1)
    all_widths = base_width * 2**np.arange(0, number_of_widths+1)

    error_names = ["Prediction error",
                  "Error mean",
                  "Error variance",
                  "Wasserstein"]


    prediction_errors = np.zeros((len(depths), len(widths)))
    wasserstein_errors = np.zeros((len(depths), len(widths)))
    mean_errors = np.zeros((len(depths), len(widths)))
    variance_errors = np.zeros((len(depths), len(widths)))
    selection_errors = np.zeros((len(depths), len(widths)))
    for (n,depth) in enumerate(depths):
        for (m,width) in enumerate(widths):
            print("Config {} x {} ([{} x {}] / [{} x {}])".format(depths[n], widths[m], n, m, len(depths), len(widths)))
            seed_random_number(random_seed)
            depth = int(depth)
            width = int(width)
            network_model = [width for k in range(depth)]
            network_model.append(1)

            title='{}_{}_{}' .format (base_title, depth, width)

            network_information.train_size = train_size
            network_information.batch_size = train_size
            network_information.validation_size = train_size
            network_information.network = network_model
            output_information.enable_plotting = False
            showAndSave.silent = True
            print_comparison_table.silent = True

            start_all_training = time.time()
            with RedirectStdStreamsToNull():
                run_function(network_information, output_information)
            end_all_training = time.time()

            duration = end_all_training - start_all_training
            print("Training and postprocessing took: {} seconds ({} minutes) ({} hours)". format(duration, duration/60, duration/60/60))

            prediction_errors[n, m] = output_information.prediction_error[2]

            mean_errors[n,m] = output_information.stat_error['mean']
            variance_errors[n,m] = output_information.stat_error['var']
            wasserstein_errors[n,m] = output_information.stat_error['wasserstein']
            selection_errors[n,m] = output_information.selection_error

    errors_map = {"Prediction error" : prediction_errors,
                  "Error mean" : mean_errors,
                  "Error variance" : variance_errors,
                  "Wasserstein" : wasserstein_errors,
                  "Selection error (%s)" % network_information.selection : selection_errors}

    all_errors_map = {}
    for k in errors_map.keys():
        all_errors_map[k] = errors_map[k]
    # doing this the safe way

    for error_name in errors_map.keys():
        if only_selection and 'Selection error' not in error_name:
            continue
        showAndSave.silent = False
        w,d = np.meshgrid(all_widths, all_depths)

        plt.pcolormesh(d, w, all_errors_map[error_name])

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Depth")
        plt.ylabel("Width")
        plt.colorbar()
        plt.title("{} with {} samples\n{}".format(error_name, train_size, base_title))
        showAndSave.prefix='%s_network_%s' % (base_title, train_size)
        np.save('results/' + showAndSave.prefix + '_{}.npy'.format(error_name.replace(" ", "")), all_errors_map[error_name])

        showAndSave(error_name.replace(" ", ""))

        print_memory_usage()


    return selection_errors, errors_map
