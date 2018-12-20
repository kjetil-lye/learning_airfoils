
import json
import post_process_hyperparameters
import copy

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

            print("Possible configurations for {target}".format(target=target))


            for close_configuration in close_configurations:
                print("\t{}: {}".format(config_to_str(close_configuration), get_values_of_config(close_configuration, targets_to_store)))

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
