import numpy as np
from mpi4py import MPI
import tensorflow as tf
from machine_learning import random_seed, seed_random_number, print_memory_usage
from plot_info import *
from print_table import *
import time
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def find_best_network_size(*, network_information,
    output_information,
    training_sizes,
    run_function,
    number_of_depths,
    number_of_widths,
    base_title):


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if int(np.sqrt(size)) != np.sqrt(size):
        raise Exception("number of MPI processors needs to be a square number")
    number_of_procs_for_width = int(np.sqrt(size))
    number_of_procs_for_depth = int(np.sqrt(size))
    number_of_widths_per_proc = number_of_widths // number_of_procs_for_width
    number_of_depths_per_proc = number_of_depths // number_of_procs_for_depth

    start_width = number_of_widths_per_proc*(rank % number_of_procs_for_width)
    start_depth = number_of_depths_per_proc*(rank // number_of_procs_for_width)

    base_width=6
    base_depth=4
    widths = base_width*2**np.arange(start_width, start_width + number_of_widths_per_proc)
    depths = base_depth*2**np.arange(start_depth, start_depth + number_of_depths_per_proc)


    print(widths)
    print(depths)


    all_depths = base_depth * 2**np.arange(0, number_of_depths+1)
    all_widths = base_width * 2**np.arange(0, number_of_widths+1)
    print(all_depths)
    print(all_widths)
    error_names = ["Prediction error",
                  "Error mean",
                  "Error variance",
                  "Wasserstein"]
    best_errors = {}

    for error_name in error_names:
        best_errors[error_name] = []

    for train_size in training_sizes:
        prediction_errors = np.zeros((len(depths), len(widths)))
        wasserstein_errors = np.zeros((len(depths), len(widths)))
        mean_errors = np.zeros((len(depths), len(widths)))
        variance_errors = np.zeros((len(depths), len(widths)))
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
                run_function(network_information, output_information)
                end_all_training = time.time()

                duration = end_all_training - start_all_training
                print("Training and postprocessing took: {} seconds ({} minutes) ({} hours)". format(duration, duration/60, duration/60/60))

                prediction_errors[n, m] = output_information.prediction_error[2]

                mean_errors[n,m] = output_information.stat_error['mean']
                variance_errors[n,m] = output_information.stat_error['var']
                wasserstein_errors[n,m] = output_information.stat_error['wasserstein']



        errors_map = {"Prediction error" : prediction_errors,
                      "Error mean" : mean_errors,
                      "Error variance" : variance_errors,
                      "Wasserstein" : wasserstein_errors}

        best_errors = {}
        all_errors_map = {}
        for k in errors_map.keys():
            best_errors[k] = []
            all_errors_map[k] = None
        # doing this the safe way

        for error_name in errors_map.keys():
            comm.barrier()

            if rank == 0:
                all_errors_map[error_name] = np.zeros((number_of_depths, number_of_widths))

            for n in range(len(depths)):
                for m in range(len(widths)):

                    # This could probably have been done with one gather, but doing it
                    # the stupid way to make sure it is correct
                    sub_errors = comm.gather([depths[n], widths[m], errors_map[error_name][n,m]],
                        root=0)

                    if rank == 0:
                        for error_pair in sub_errors:
                            depth = error_pair[0]
                            width = error_pair[1]
                            error = error_pair[2]

                            i = np.where(all_depths == depth)
                            j = np.where(all_widths == width)

                            all_errors_map[error_name][i,j] = error

            if rank == 0:
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
                np.save(showAndSave.prefix + '_{}.npy'.format(error_name.replace(" ", "")), all_errors_map[error_name])

                showAndSave(error_name.replace(" ", ""))
                best_errors[error_name].append(np.amin(all_errors_map[error_name]))
                print_memory_usage()
    for error_name in errors_map.keys():
        if rank == 0:
            showAndSave.silent = False
            plt.loglog(training_sizes, best_errors[error_name], '-o')
            plt.xlabel('Training size')
            plt.ylabel("Best {}".format(error_name))

            showAndSave.prefix = '{}_{}_network_' % (base_title, train_size)
            showAndSave('best_network')
            np.save(showAndSave.prefix + "best_network{}.npy".format(error_name.replace(" ", "_")), np.array(best_predictions))
