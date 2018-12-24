
import json
import post_process_hyperparameters
import copy
import numpy as np
import print_table
def get_dict_path(dictionary, path):
    split = path.split('.')
    for s in split:
        dictionary = dictionary[s]
    return dictionary

def find_minimum(configurations, targets, comparator):
    best_values = {}

    best_configurations = {}
    for m in configurations:
        for k in targets:

            if k not in best_values or comparator(best_values[k], get_dict_path(m, k)):
                best_values[k] = get_dict_path(m, k)

                best_configurations[k] = copy.deepcopy(m)

    return best_values, best_configurations


def find_close_configurations(measurements, key_to_compare, value_of_key, distance):
    compatible_measurements = []
    for m in measurements:
        value = get_dict_path(m, key_to_compare)
        if abs(value-value_of_key) <= distance:
            compatible_measurements.append(copy.deepcopy(m))
    return compatible_measurements

def regularization_to_str(regularization):
    if regularization is None or regularization == "None":
        return "None"
    else:
        return "l1_{l1}_l2_{l2}".format(l1=regularization['l1'], l2=regularization['l2'])

def config_to_str(configuration):
    return "{optimizer}_{loss}_{selection}_{regularizer}".format(
        optimizer = post_process_hyperparameters.get_optimizer(configuration),
        loss = post_process_hyperparameters.get_loss(configuration),
        selection = post_process_hyperparameters.get_selection(configuration),
        regularizer = regularization_to_str(post_process_hyperparameters.get_regularization(configuration))
    )

def get_values_of_config(configuration, targets):
    values = {}

    for target in targets:
        target_cleaned = target.split(".")[-1]
        values[target_cleaned] = get_dict_path(configuration, target)
    return ", ".join([str(values[k]) for k in values.keys()])

def find_intersections(filenames, data_source, convergence_rate):
    targets = [
        'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_error_cut'.format(data_source=data_source),
        'results.best_network.algorithms.{data_source}.ml.ordinary.prediction_l2_relative'.format(data_source=data_source),

    ]

    allowed_distances = {
        'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_error_cut'.format(data_source=data_source) : 2**(-11),
        'results.best_network.algorithms.{data_source}.ml.ordinary.prediction_l2_relative'.format(data_source=data_source) : 0.025
    }

    targets_to_store = copy.deepcopy(targets)
    targets_to_store.append('results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_raw'.format(data_source=data_source))

    all_intersections = None
    all_errors = {}

    data = {}
    for functional in filenames.keys():
        filename = filenames[functional]
        with open(filename, 'r') as f:
            json_content = json.load(f)
            # only look at best performing

            post_process_hyperparameters.fix_bilevel(json_content)
            post_process_hyperparameters.add_wasserstein_speedup(json_content, convergence_rate)
            json_content = post_process_hyperparameters.filter_configs(json_content, onlys={"settings.selection_type":["Best performing"],
                "settings.train_size":[128]})
            data[functional] = copy.deepcopy(json_content)
        minimum_values, minimum_configuration = find_minimum(data[functional]['configurations'], targets, lambda x, y: x > y)

        intersected_configurations = None
        for target in targets:
            close_configurations = find_close_configurations(data[functional]['configurations'],
                target, minimum_values[target], allowed_distances[target])

            # Only look at those with more than 1.4 in wasserstein Speedup
            for config in close_configurations:
                if get_dict_path(config, 'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_raw'.format(data_source=data_source)) < 1.4:
                    close_configurations.remove(config)

            # print("Possible configurations for {target}".format(target=target))
            #
            #
            for close_configuration in close_configurations:
            #     print("\t{}: {}".format(config_to_str(close_configuration), get_values_of_config(close_configuration, targets_to_store)))
            #
                 if config_to_str(close_configuration) not in all_errors.keys():
                     all_errors[config_to_str(close_configuration)] = {}
                 all_errors[config_to_str(close_configuration)][functional] = get_values_of_config(close_configuration, targets_to_store)



            configurations_as_str = [config_to_str(conf) for conf in close_configurations]

            if intersected_configurations is not None:
                intersected_configurations = intersected_configurations.intersection(set(configurations_as_str))
            else:
                intersected_configurations = set(configurations_as_str)
        if all_intersections is None:
            all_intersections = intersected_configurations
        else:
            all_intersections=all_intersections.intersection(intersected_configurations)

    print()
    print()
    print("Possible intersections:")
    for config in all_intersections:
        print("\t{}".format(config))
        for func in all_errors[config]:
            print("\t\t{}: {}".format(func, all_errors[config][func]))
        print()
        print()
def get_retraining_values(config, key):
    values = []
    for k in get_dict_path(config,'results.retrainings').keys():
        new_key = key.replace('best_network', 'retrainings.{}'.format(k))
        values.append(get_dict_path(config, new_key))
    return values
def get_worst_retrain_value(config, key):
    operator = np.max

    if 'speedup' in key:
        operator = np.min

    return operator(get_retraining_values(config, key))


def print_configurations_to_file(filename, configurations):

    configurations_postprocssed = []
    values_to_print = [
        'settings.selction',
        'settings.regularizer',
        'settings.optimizer',
        'settings.loss'
    ]
    for config in configurations:
        new_config = {}
        for k in values_to_print:
            new_config[k] = get_dict_path(config, k)
        configurations_postprocssed.append(new_config)
    with open(filename, 'w') as outfile:
        json.dump({"configurations": configurations_postprocssed}, outfile)
def find_intersections_acceptable(filenames, data_source, convergence_rate, get_value= lambda x, key: get_dict_path(x, key),
 max_prediction=0.05, min_speedup=2, print_filename=None, table_filename = None):

    functionals = [k for k in filenames.keys()]
    targets = [
        'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_raw'.format(data_source=data_source),
        'results.best_network.algorithms.{data_source}.ml.ordinary.prediction_l2_relative'.format(data_source=data_source)

    ]

    accepted = {
        'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_raw'.format(data_source=data_source) : lambda x: x>min_speedup,
        'results.best_network.algorithms.{data_source}.ml.ordinary.prediction_l2_relative'.format(data_source=data_source) : lambda x: x<max_prediction
    }

    targets_to_store = copy.deepcopy(targets)
    targets_to_store.append('results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_real'.format(data_source=data_source))
    stats = {
        'min' : lambda x, target: np.min(get_retraining_values(x, target)),
        'selected' : lambda x, target: get_dict_path(x, target),
        'std' : lambda x, target: np.sqrt(np.var(get_retraining_values(x, target))),
        'mean' : lambda x, target: np.mean(get_retraining_values(x, target)),
        'max' : lambda x, target: np.max(get_retraining_values(x, target))
    }
    all_intersections = None
    all_errors = {}
    all_stats = {}
    actual_configurations = {}

    data = {}

    for stat in stats.keys():
        all_stats[stat] = {}
    for functional in filenames.keys():
        filename = filenames[functional]
        with open(filename, 'r') as f:
            json_content = json.load(f)
            # only look at best performing

            post_process_hyperparameters.fix_bilevel(json_content)
            post_process_hyperparameters.add_wasserstein_speedup(json_content, convergence_rate)
            json_content = post_process_hyperparameters.filter_configs(json_content, onlys={"settings.selection_type":["Best performing"],
                "settings.train_size":[128]})
            data[functional] = copy.deepcopy(json_content)

        intersected_configurations = None
        for target in targets:
            close_configurations = []
            for config in data[functional]['configurations']:
                value = get_value(config, target)
                if accepted[target](value):
                    close_configurations.append(config)


            # print("Possible configurations for {target}".format(target=target))
            #
            #
            for close_configuration in close_configurations:
            #     print("\t{}: {}".format(config_to_str(close_configuration), get_values_of_config(close_configuration, targets_to_store)))
            #
                 if config_to_str(close_configuration) not in all_errors.keys():
                     all_errors[config_to_str(close_configuration)] = {}
                 all_errors[config_to_str(close_configuration)][functional] = get_values_of_config(close_configuration, targets_to_store)

                 for stat in all_stats.keys():
                     if config_to_str(close_configuration) not in all_stats[stat].keys():
                          all_stats[stat][config_to_str(close_configuration)] = {}
                     if functional not in  all_stats[stat][config_to_str(close_configuration)].keys():
                          all_stats[stat][config_to_str(close_configuration)][functional] = {}
                     for target_store in targets_to_store:
                         all_stats[stat][config_to_str(close_configuration)][functional][target_store] = stats[stat](close_configuration, target_store)
                 actual_configurations[config_to_str(close_configuration)] = copy.deepcopy(close_configuration)

            configurations_as_str = [config_to_str(conf) for conf in close_configurations]

            if intersected_configurations is not None:
                intersected_configurations = intersected_configurations.intersection(set(configurations_as_str))
            else:
                intersected_configurations = set(configurations_as_str)
        if all_intersections is None:
            all_intersections = intersected_configurations
        else:
            all_intersections=all_intersections.intersection(intersected_configurations)

    print()
    print()
    print("Possible intersections:")
    for config in all_intersections:
        print("\t{}".format(config))
        for func in all_errors[config]:
            print("\t\t{}: {}".format(func, all_errors[config][func]))
        print()
        print()

    if print_filename is not None:
        print_configurations_to_file(print_filename, [actual_configurations[k] for k in all_intersections])

    if table_filename is not None:
        print_table_from_config(table_filename, [actual_configurations[k] for k in all_intersections], targets_to_store, functionals, all_stats)

def regularization_to_str_pretty(regularization):
    if regularization is None or regularization == "None":
        return "None"
    else:
        return "l1_{l1:.1e}_l2_{l2:.1e}".format(l1=regularization['l1'], l2=regularization['l2'])

def pretty_loss(loss):
    if loss == 'mean_squared_error':
        return 'mse'
    elif loss == 'mean_absolute_error':
        return 'mae'
    else:
        return loss

def pretty_print_config(configuration):

    return "{optimizer}_{loss}_{selection}_{regularizer}".format(
        optimizer = post_process_hyperparameters.get_optimizer(configuration)[0],
        loss = pretty_loss(post_process_hyperparameters.get_loss(configuration)),
        selection = post_process_hyperparameters.get_selection(configuration)[0],
        regularizer = regularization_to_str_pretty(post_process_hyperparameters.get_regularization(configuration))
    )
def regularization_to_row(regularization):
    if regularization == "None":
        return regularization_to_row({"l1":0, "l2":0})
    else:
        return ['{:.1e}'.format(regularization['l1']),
                '{:.1e}'.format(regularization['l2'])]

def config_to_row(configuration):
    return [
        post_process_hyperparameters.get_optimizer(configuration),
        pretty_loss(post_process_hyperparameters.get_loss(configuration)),
        post_process_hyperparameters.get_selection(configuration),
        *regularization_to_row(post_process_hyperparameters.get_regularization(configuration))
    ]
def config_header_row():
    return ["Optimizer", "Loss", "Selection", "L1reg", "L2reg"]

def print_table_from_config(filename, configurations, targets, functionals, all_stats):
    stats = {
        'min' : lambda x: np.min(x),
        'std' : lambda x: np.sqrt(np.var(x)),
        'mean' : lambda x: np.mean(x),
        'max' : lambda x: np.max(x)
    }

    stats_to_print = [
        "selected",
        "min",
        "mean",
        "max",
        "std"
    ]
    targets_short_names = [t.split('.')[-1] for t in targets]
    for functional in functionals:
        for target, short_target in zip(targets, targets_short_names):
            table_builder = print_table.TableBuilder()



            lower_header = config_header_row()

            for stat in stats_to_print:
                name = "{}".format(stat)
                lower_header.append(name)


            table_builder.set_header(lower_header)
            values = {}
            for k in stats_to_print:
                values[k] = []
            new_rows = []
            for config in configurations:
                row = config_to_row(config)

                for stat in stats_to_print:
                    row.append('{:.3e}'.format(all_stats[stat][config_to_str(config)][functional][target]))

                    values[stat].append(all_stats[stat][config_to_str(config)][functional][target])
                new_rows.append(row)

            new_rows = sorted(new_rows, key=lambda row: float(row[5]))
            if 'speedup' in target:
                new_rows.reverse()
            for row in new_rows:
                table_builder.add_row(row)
            for stat in stats.keys():
                row = []
                for k in config_header_row()[:2]:
                    row.append('--')

                row.append('{stat}'.format(stat=stat))

                for k in config_header_row()[3:]:
                    row.append('--')

                for stat2 in stats_to_print:
                    row.append('{:.3e}'.format(stats[stat](values[stat2])))
                table_builder.add_row(row)


            table_builder.set_title("Sensitivity for {functional} ({target}) for best configurations".format(target=short_target, functional=functional))
            table_builder.print_table("{filename}_{functional}_{target}".format(filename=filename, functional=functional, target=short_target))
    for target, short_target in zip(targets, targets_short_names):
        table_builder = print_table.TableBuilder()
        lower_header = config_header_row()

        for functional in functionals:
            name = "{}".format(functional)
            lower_header.append(name)


        table_builder.set_header(lower_header)
        values = {}
        for k in functionals:
            values[k] = []

        new_rows = []
        for config in configurations:
            row = config_to_row(config)

            for functional in functionals:
                row.append('{:.3e}'.format(all_stats['selected'][config_to_str(config)][functional][target]))

                values[stat].append(all_stats['selected'][config_to_str(config)][functional][target])
            new_rows.append(row)

        new_rows = sorted(new_rows, key=lambda row: float(row[5]))
        if 'speedup' in target:
            new_rows.reverse()
        for row in new_rows:
            table_builder.add_row(row)
        for stat in stats.keys():
            row = []
            for k in config_header_row()[:2]:
                row.append('--')

            row.append('{stat}'.format(stat=stat))

            for k in config_header_row()[3:]:
                row.append('--')

            for stat2 in stats_to_print:
                row.append('{:.3e}'.format(stats[stat](values[stat2])))
            table_builder.add_row(row)


        table_builder.set_title("Best retrained values for best configurations for {target}".format(target=short_target, functional=functional))
        table_builder.print_table("{filename}_selected_{target}".format(filename=filename, functional=functional, target=short_target))
