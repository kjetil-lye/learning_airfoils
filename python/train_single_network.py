import numpy as np
import tensorflow as tf
from machine_learning import NetworkInformation, OutputInformation, get_network_and_postprocess, random_seed, seed_random_number, print_memory_usage, Tables
from plot_info import *
from print_table import *
import time
import keras
import network_parameters
import json

def train_single_network(*, parameters, samples, base_title, network, epochs, large_integration_points = None, sampling_method='QMC'):
    train_sizes = network_parameters.get_training_sizes()

    optimizers = network_parameters.get_optimizers()

    losses = network_parameters.get_losses()
    selections = network_parameters.get_selections()


    for selection_type in selections.keys():

        display(HTML("<h1>%s</h1>" % selection_type))
        for selection in selections[selection_type]:
            for optimizer in optimizers.keys():
                for loss in losses:
                    display(HTML("<h1>{} with {}</h1>".format(optimizer, loss)))

                    for train_size in train_sizes:

                        regularizations = network_parameters.get_regularizations(train_size)
                        for regularization in regularizations:
                            regularization_name = "No regularization"
                            if regularization is not None:
                                if regularization.l2 > 0:
                                    regularization_name = "l2 (%f)" % regularization.l2
                                else:
                                    regularization_name = "l1 (%f)" % regularization.l1
                            tables = Tables.make_default()
                            display(HTML("<h4>%s</h4>" % regularization_name))
                            seed_random_number(random_seed)

                            showAndSave.silent = False
                            print_comparison_table.silent = False
                            title = '%s\nRegularization:%s\nSelection Type: %s, Selection criterion: %s\nLoss function: %s, Optimizer: %s, Train size: %d' % (base_title, regularization_name, selection_type, selection, loss, optimizer, train_size)
                            network_information = NetworkInformation(optimizer=optimizers[optimizer], epochs=epochs,
                                                                     network=network, train_size=train_size,
                                                                     validation_size=train_size,
                                                                     loss=loss,
                                                                     large_integration_points=large_integration_points,
                                                                     selection=selection, tries=5,
                                                                     kernel_regularizer = regularization)

                            output_information = OutputInformation(tables=tables, title=title,
                                                                  short_title=title, enable_plotting=True,
                                                                  sampling_method=sampling_method)
                            showAndSave.prefix = '%s_%s_%s_%s_%s_%s_%d' % (only_alphanum(base_title),
                                only_alphanum(regularization_name), only_alphanum(selection_type),
                                only_alphanum(selection), loss, only_alphanum(optimizer), train_size)

                            get_network_and_postprocess(parameters, samples, network_information = network_information,
                                output_information = output_information)

                            prediction_error = output_information.prediction_error[2]

                            mean_error= copy.deepcopy(output_information.stat_error['mean'])
                            variance_error = copy.deepcopy(output_information.stat_error['var'])
                            wasserstein_error = copy.deepcopy(output_information.stat_error['wasserstein'])
                            selection_error = copy.deepcopy(output_information.selection_error)

                            error_map = {"main_error" : mean_error,
                                        "variance_error" : variance_error,
                                        "wasserstein_error" : wasserstein_error,
                                        "selection_error" : selection_error,
                                        "prediction_error" : prediction_error}

                            with open('results/' + showAndSave.prefix + '_errors.json', 'w') as out:
                                json.dump(error_map, out)

                            print(json.dumps(error_map))
                            console_log(json.dumps(error_map))
                            tables.write_tables()
