import numpy as np
import tensorflow as tf
from machine_learning import NetworkInformation, OutputInformation, get_network_and_postprocess, random_seed, seed_random_number, print_memory_usage, Tables
from plot_info import *
from print_table import *
import time
import keras
import network_parameters
import json

def train_single_network(*, parameters, samples, base_title, network,
    large_integration_points = None,
    sampling_method='QMC',
    base_config = None,
    monte_carlo_parameters = None,
    monte_carlo_values = None,
    load_network_weights = False):

    losses = None
    optimizers_to_choose = None
    selection_to_choose = None
    regularizations_to_choose = None

    if base_config is not None:
        losses = [base_config['loss']]
        optimizers_to_choose = base_config['optimizer']
        selection_to_choose = base_config['selection']
        regularizations_to_choose = [base_config['regularization']]


    train_sizes = network_parameters.get_training_sizes()


    optimizers = network_parameters.get_optimizers()
    losses = losses or network_parameters.get_losses()
    selections = network_parameters.get_selections()



    for selection_type in selections.keys():

        display(HTML("<h1>%s</h1>" % selection_type))
        for selection in selections[selection_type]:
            if selection_to_choose and selection != selection_to_choose:
                continue
            for optimizer in optimizers.keys():
                if optimizers_to_choose and optimizer != optimizers_to_choose:
                    continue
                for loss in losses:
                    display(HTML("<h1>{} with {}</h1>".format(optimizer, loss)))

                    for train_size in train_sizes:

                        regularizations = regularizations_to_choose or network_parameters.get_regularizations(train_size)
                        for regularization in regularizations:
                            regularization_name = "No regularization"
                            if regularization is not None:
                                if regularization.l2 > 0:
                                    regularization_name = "l2 (%.4e)" % regularization.l2
                                else:
                                    regularization_name = "l1 (%.4e)" % regularization.l1

                            learning_rates = network_parameters.get_learning_rates()
                            for learning_rate in learning_rates:
                                epochs = network_parameters.get_epochs()
                                for epoch in epochs:

                                    tables = Tables.make_default()
                                    display(HTML("<h4>%s</h4>" % regularization_name))
                                    seed_random_number(random_seed)

                                    showAndSave.silent = False
                                    print_comparison_table.silent = False
                                    title = '%s\nRegularization:%s\nSelection Type: %s, Selection criterion: %s\nLoss function: %s, Optimizer: %s, Train size: %d\nEpochs: %d, learning rate: %f' % (base_title, regularization_name, selection_type, selection, loss, optimizer, train_size, epoch, learning_rate)
                                    network_information = NetworkInformation(optimizer=optimizers[optimizer],
                                                                             network=network, train_size=train_size,
                                                                             validation_size=train_size,
                                                                             loss=loss,
                                                                             learning_rate=learning_rate,
                                                                             epochs=epoch,
                                                                             large_integration_points=large_integration_points,
                                                                             selection=selection, tries=5,
                                                                             kernel_regularizer = regularization,
                                                                             monte_carlo_values= monte_carlo_values,
                                                                             monte_carlo_parameters = monte_carlo_parameters)


                                    output_information = OutputInformation(tables=tables, title=title,
                                                                          short_title=title, enable_plotting=True,
                                                                          sampling_method=sampling_method)
                                    showAndSave.prefix = '%s_%s_%s_%s_%s_%s_%d_%s_%s' % (only_alphanum(base_title),
                                        only_alphanum(regularization_name), only_alphanum(selection_type),
                                        only_alphanum(selection), loss, only_alphanum(optimizer), train_size,
                                        str(epoch),
                                        only_alphanum("{}".format(learning_rate)))
                                    print("Training: {}".format(title))

                                    with RedirectStdStreamsToNull() as _:
                                        if load_network_weights:
                                            network_weight_filename ='results/' + showAndSave.prefix +  'model.h5'
                                            network_structure_filename = 'results/' + showAndSave.prefix +  'model.json'


                                            network_information.network_weights_filename = network_weight_filename
                                            network_information.network_structure_filename = network_structure_filename

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


def compute_for_all_in_json(json_file, *, parameters, samples, base_title, network,
    large_integration_points = None,
    sampling_method='QMC',
    monte_carlo_values = None,
    monte_carlo_parameters = None,
    load_network_weights = False):

    with open(json_file) as infile:
        configurations = json.load(infile)

        for configuration_name in configurations.keys():
            config = configurations[configuration_name]

            if config['regularization'] == 'None':
                config['regularization'] = None
            else:
                if config['regularization']['l1'] > 0:
                    config['regularization'] = keras.regularizers.l1(config['regularization']['l1'] )
                else:
                    config['regularization'] = keras.regularizers.l2(config['regularization']['l2'] )

            display(HTML("<h1>{}</h1>".format(configuration_name)))

            train_single_network(parameters=parameters,
                                 samples=samples,
                                 base_title=base_title,
                                 network=network,
                                 large_integration_points=large_integration_points,
                                 sampling_method=sampling_method,
                                 base_config = config,
                                 monte_carlo_values = monte_carlo_values,
                                 monte_carlo_parameters = monte_carlo_parameters,
                                 load_network_weights = load_network_weights)
