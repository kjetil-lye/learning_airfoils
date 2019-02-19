#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../python')
import print_table
import plot_info
import json
from IPython.core.display import display, HTML
import copy
import pprint
import re
import os
import bz2


def fix_bilevel(configuration):
    """
    In an earlier verison (4f65635f5b32b842b8b5c80f9978520c85545b25) there was an error in how
    the error (yes) of bilevel was calculated. This fixes this, and adds relative errors
    """

    for config in configuration['configurations']:
        sources = [config['results']['best_network']['algorithms']]
        sources.extend([config['results']['retrainings'][k]['algorithms'] for k in config['results']['retrainings'].keys()])
        sources.extend([config['network_sizes'][k]['results']['best_network']['algorithms'] for k in range(len(config['network_sizes']))])
        basemean = config['results']['best_network']['reference_sampling_error']['mean']
        basevar = config['results']['best_network']['reference_sampling_error']['var']

        basemeanmlmc = config['results']['best_network']['base_sampling_error']['mean_bilevel']
        basevarmlmc = config['results']['best_network']['base_sampling_error']['var_bilevel']

        config['results']['best_network']['base_sampling_error']['mean_bilevel_error'] = abs(basemean-basemeanmlmc)
        config['results']['best_network']['base_sampling_error']['mean_bilevel_error_relative'] = abs(basemean-basemeanmlmc)/abs(basemean)

        config['results']['best_network']['base_sampling_error']['var_bilevel_error'] = abs(basevar-basevarmlmc)
        config['results']['best_network']['base_sampling_error']['var_bilevel_error_relative'] = abs(basevar-basevarmlmc)/abs(basevar)


        for source in sources:
            for algorithm in source.keys():
                for fit in source[algorithm].keys():
                    for tactic in source[algorithm][fit].keys():


                        mlmc_mean = source[algorithm][fit][tactic]['mean_bilevel']
                        source[algorithm][fit][tactic]['mean_bilevel_error'] =  abs(mlmc_mean-basemean)
                        source[algorithm][fit][tactic]['mean_bilevel_error_relative'] =abs(mlmc_mean-basemean)/abs(basemean)


                        mlmc_var = config['results']['best_network']['algorithms'][algorithm][fit][tactic]['var_bilevel']
                        source[algorithm][fit][tactic]['var_bilevel_error'] = abs(mlmc_var-basevar)
                        source[algorithm][fit][tactic]['var_bilevel_error_relative'] = abs(mlmc_var-basevar)/abs(basevar)


def add_wasserstein_speedup(configuration, convergence_rate):
    """
    Adds the wasserstein speedup to the data.
    """

    for config in configuration['configurations']:
        sources = [config['results']['best_network']['algorithms']]
        sources.extend([config['results']['retrainings'][k]['algorithms'] for k in config['results']['retrainings'].keys()])
        sources.extend([config['network_sizes'][k]['results']['best_network']['algorithms'] for k in range(len(config['network_sizes']))])
        # Speedup is always one when we compare to ourself:
        config['results']['best_network']['base_sampling_error']['wasserstein_speedup'] = 1

        #####
        for source in sources:
            for algorithm in source.keys():
                for fit in source[algorithm].keys():

                    for tactic in source[algorithm][fit].keys():
                        base = config['results']['best_network']['base_sampling_error']['wasserstein_error_cut']
                        wasserstein = source[algorithm][fit][tactic]['wasserstein_error_cut']
                        source[algorithm][fit][tactic]['wasserstein_speedup_raw'] = base/wasserstein

                        source[algorithm][fit][tactic]['wasserstein_speedup_real'] = (base/wasserstein)**(1.0/convergence_rate)

def regularization_to_str(regularization):
    if regularization is None or regularization == "None":
        return "None"
    else:
        return "l1_{l1}_l2_{l2}".format(l1=regularization['l1'], l2=regularization['l2'])

def config_to_str_from_json(configuration):
    return "{optimizer}_{loss}_{selection}_{regularizer}_{train_size}".format(
        optimizer = get_optimizer(configuration),
        loss = get_loss(configuration),
        selection = get_selection(configuration),
        train_size = get_dict_path(configuration, 'settings.train_size'),
        regularizer = regularization_to_str(get_regularization(configuration))
    )
def config_to_str(configuration):
    return "{optimizer}_{loss}_{selection}_{regularizer}".format(
        optimizer = configuration['settings.optimizer'],
        loss = configuration['settings.loss'],
        selection = configuration['settings.selction'],
        regularizer = configuration['settings.regularizer']
    )

class FilterFromConfiguration(object):
    def __init__(self, configuration, name):
        self.configuration = copy.deepcopy(configuration)
        self.name = name

    def __call__(self, configuration_on_trial):
        for k in self.configuration.keys():
            value = get_dict_path(configuration_on_trial, k)
            value_required = self.configuration[k]

            if value != value_required:
                return False

        return True

def get_filters_from_file(filename):
    filters = []

    with open(filename) as infile:
        json_content = json.load(infile)

        for configuration in json_content['configurations']:
            filters.append(FilterFromConfiguration(configuration,
                config_to_str(configuration)))
    return filters


def load_all_configurations(filename):
    if filename.endswith('.bz2'):
        with bz2.BZ2File(filename) as infile:
            return json.loads(infile.read().decode('utf-8'))
    else:
        with open(filename) as infile:
            return json.load(infile)

def plot_all(filenames, convergence_rate, latex_out, data_source='QMC_from_data', only_network_sizes=False):
    functionals = [k for k in filenames.keys()]
    data = {}

    for functional in functionals:
        data[functional] = []

        filename =filenames[functional]

        json_content = load_all_configurations(filename)
        fix_bilevel(json_content)
        add_wasserstein_speedup(json_content, convergence_rate)
        data[functional] = copy.deepcopy(json_content)

        targets_to_store = [
            'results.best_network.algorithms.{data_source}.ml.replace.wasserstein_speedup_raw'.format(data_source=data_source),
            'results.best_network.algorithms.{data_source}.ml.ordinary.prediction_l2_relative'.format(data_source=data_source)

        ]
        found_configs = {}
        # check for uniqueness
        for config in  data[functional]['configurations']:
            if config_to_str_from_json(config) in found_configs.keys():
                if [get_dict_path(config, tar) for tar in targets_to_store] != [get_dict_path(found_configs[config_to_str_from_json(config)], tar) for tar in targets_to_store]:
                    raise Exception("Same config appearead twice: {} ({})\n\n{}\n\n{}\n\n{}\n{}".format(config_to_str_from_json(config), filename, str(config['settings']), str(found_configs[config_to_str_from_json(config)]['settings']),
                        [get_dict_path(config, tar) for tar in targets_to_store],
                            [get_dict_path(found_configs[config_to_str_from_json(config)], tar) for tar in targets_to_store]
                            ))
                else:
                    data[functional]['configurations'].remove(config)
            found_configs[config_to_str_from_json(config)] = config

    onlys = {
        "" : {},
        "Best performing": {"settings.selection_type":["Best performing"]},
        "Emperically optimal" :{"settings.selection_type":["Emperically optimal"]},

        "Best performing mean_m2": {"settings.selection_type":["Best performing"], "settings.loss": ["mean_m2"]},
        "Emperically optimal mean_m2" :{"settings.selection_type":["Emperically optimal"], "settings.loss": ["mean_m2"]},

        "Best performing mse mae": {"settings.selection_type":["Best performing"], "settings.loss": ["mean_squared_error", "mean_absolute_error"]},
        "Emperically optimal mse mae" :{"settings.selection_type":["Emperically optimal"], "settings.loss": ["mean_squared_error", "mean_absolute_error"]},


    }


    filters = {
        'All configurations' : lambda x: True,

        'Only Adam with ($L^2$ and no regularization) or ($L^1$)' : only_adam_and_no_regularization_for_mse,
        'Only Adam with ($L^2$ and no regularization) or ($L^1$ with regularization)' : only_adam_and_no_regularization_for_mse_and_reg_for_l1,
        'Only Adam with ($L^2$ and low regularization) or ($L^1$ with regularization)' : only_adam_and_low_regularization_for_mse_and_reg_for_l1
    }

    filters_single = {}
    #for f in [best_configuration_1, best_configuration_2, best_configuration_3, best_configuration_4]:
    #    filters_single[f.__doc__] = f
    #if not os.path.exists('acceptable.json'):
    #    raise Exception("You should run the notebook Intersection.ipynb to generate the acceptable.json file!")
    #for f in get_filters_from_file('acceptable.json'):
    #    filters_single[f.name] = f


    latex = LatexWithAllPlots()
    plot_info.savePlot.callback = latex
    print_table.print_comparison_table.callback = lambda x, title: latex.add_table(x, title)

    def heading(level, content):

        if level == 0:
            html_code = 'h1'
            latex_code = 'section'
        elif level == 1:
            html_code = 'h2'
            latex_code = 'subsection'

        else:
            html_code = 'h3'
            latex_code = 'subsubsection'
        display(HTML("<{ht}>{title}<{ht}>".format(ht=html_code, title=content)))
        latex.text+= "\\{latex_code}{{{title}}}\n\n".format(latex_code=latex_code, title=content)

    comparisons = [
        ["mean_m2", get_only_mean_m2, "MSE and MAE", complement(get_only_mean_m2)],
        ["mean_m2", get_only_mean_m2, "MSE", get_only_mse],
        ["mean_m2", get_only_mean_m2, "MAE", get_only_mae],
        ["Good Adam", only_adam_and_low_regularization_for_mse_and_reg_for_l1, "Bad Adam", and_config(get_only_adam, complement(only_adam_and_low_regularization_for_mse_and_reg_for_l1))],
        ["SGD", get_only_sgd, "Adam", get_only_adam],
        ["Good Adam MSE", and_config(get_only_mse, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam MAE", and_config(get_only_mae, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
        ["Good Adam L1 reg", and_config(only_l1_reg, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam L2 reg", and_config(only_l2_reg, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
        ["Good Adam with val", and_config(only_ray_prediction, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam with train", and_config(only_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
        ["Good Adam with val", and_config(only_ray_prediction, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam with wass-train", and_config(only_wasserstein_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
        ["Good Adam with train", and_config(only_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam with wass-train", and_config(only_wasserstein_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
        ["Good Adam with mean-train", and_config(only_mean_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1), "Good Adam with wass-train", and_config(only_wasserstein_train, only_adam_and_low_regularization_for_mse_and_reg_for_l1)],
    ]


    for functional in functionals:
        heading(0, functional)
        if not only_network_sizes:
            for comparison in comparisons:
                for only in onlys:
                    compare_two_sets(functional,
                        data1 = filter_configs(data[functional], test_functions = [comparison[1]]),
                        title1 = comparison[0],
                        data2 = filter_configs(data[functional], test_functions = [comparison[3]]),
                        title2 = comparison[2],
                        main_title=only
                        )
        for filtername in filters:
            heading(1, filtername)
            for only in onlys:
                heading(2, only)
                plot_as_training_size(functional, filter_configs(data[functional], test_functions=[filters[filtername]],
                                    onlys=onlys[only]), \
                                  filtername+" " + only, only_network_sizes=only_network_sizes)
        for filtername in filters_single:
            heading(1, filtername)
            plot_as_training_size(functional, filter_configs(data[functional], test_functions=[filters_single[filtername]]), filtername, only_network_sizes=only_network_sizes)

    with open(latex_out, 'w') as f:
        f.write(latex.get_latex())


class LatexWithAllPlots(object):
    def __init__(self):
        self.text = """
\\documentclass[a4paper]{article}
"""
        with open('../latex/header.tex') as header:
            self.text += header.read()
        self.text += """
\\begin{document}
\\tableofcontents
\\clearpage
        """
    def __call__(self, image_path, basename, title):
        title = title.replace("_", " ")
        self.text +=         """
%%%%%%%%%%%%%
% {full_path}
\\begin{{figure}}
\\InputImage{{0.8\\textwidth}}{{0.6\\textheight}}{{{image_name}}}
\\cprotect\\caption{{{title}\\\\
\\textbf{{To include:}}\\\\ \\verb|\\includegraphics[width=0.8\\textwidth]{{img/{image_name}}}|\\\\
Full path:\\
(\\verb|{full_path}|)
}}

\\end{{figure}}
\\clearpage
""".format(image_name=basename, full_path=image_path, title=title)

    def get_latex(self):
        return self.text + "\n\\end{document}"


    def add_table(self, tablefile, title):
        title=title.replace("_", " ")
        self.text +=         """
%%%%%%%%%%%%%
% {full_path}
\\begin{{table}}
\\input{{{full_path}}}
\\cprotect\\caption{{
{title}\\\\
\\textbf{{IMPORTANT NOTE: }} Check the tactic at the end of the file name.
If the tactic is "ordinary", this is the one to use for prediction error,
do NOT use any of the other tactics for prediction error.

The other tactics are: add, replace, remove.
\\textbf{{To include:}}\\\\ \\verb|\\input{{{full_path}}}|
}}

\\end{{table}}
\\clearpage
""".format(full_path=tablefile, title=title)

def generate_plot_name(error, functional, tactic, config, include_selected,
                include_retraining, include_min, include_max, include_std, include_competitor,
                include_extra_competitor,
                tactics_in_same_plot):

    basename = "training_size_{error}_{functional}_{tactic}_{config}".format(
        error=error,
        functional=functional,
        tactic=tactic,
        config=config
    )

    if include_selected:
        basename +='_sel'
    if include_retraining:
        basename += '_retra'
    if include_min:
        basename += '_min'
    if include_max:
        basename += '_max'
    if include_std:
        basename += '_std'
    if include_competitor:
        basename += '_comp'

    if include_extra_competitor:
        basename += '_compextra'


    return basename

def get_regularization_size(config):
    reg = get_regularization(config)
    if reg == "None":
        return 0
    else:
        return max(reg['l1'], reg['l2'])

def get_dict_path(dictionary, path):
    split = path.split('.')
    for s in split:
        dictionary = dictionary[s]
    return dictionary


# As a function of training size
def plot_as_training_size(functional, data, title="all configurations", only_network_sizes = False):
    if len(data['configurations']) == 0:

        print("No configurations!")

        return

    train_sizes = []

    for configuration in data['configurations']:
        train_size = int(configuration['settings']['train_size'])
        if train_size not in train_sizes:
            train_sizes.append(train_size)

    train_sizes = sorted(train_sizes)

    data_source_names = data['configurations'][0]['results']['best_network']['algorithms'].keys()
    for k in data_source_names:
        if re.match(r'^[Q]?MC_from_data$', k):
            data_source = k
            break

    sampling_method = re.search(r'(^[Q]?MC)_from_data$', data_source).group(1)
    names = {
        #"mean_error_relative" : "mean relative error",
        #"var_error_relative" : "variance relative error",
        "wasserstein_error_cut" : "Wasserstein",
        #"mean_bilevel_error_relative": "relative error bilevel mean",
        #"var_bilevel_error_relative" :"relative error bilevel variance",
        #"prediction_l1_relative": 'relative prediction error ($L^1$)',
        "prediction_l2_relative" : 'relative prediction error ($L^2$)',
        'wasserstein_speedup_raw' : 'Raw Wasserstein speedup',
        'wasserstein_speedup_real' : 'Wasserstein speedup with convergence rate',
    }

    competitor_keys = {
        "mean_error_relative" : "results.best_network.base_sampling_error.mean_error_relative",
        "var_error_relative" : "results.best_network.base_sampling_error.mean_error_relative",
        "wasserstein_error_cut" : "results.best_network.base_sampling_error.wasserstein_error_cut",
        "mean_bilevel_error_relative": "results.best_network.base_sampling_error.mean_error_relative",
        "var_bilevel_error_relative" :"results.best_network.base_sampling_error.var_error_relative",
        "prediction_l1_relative": 'results.best_network.algorithms.{data_source}.lsq.ordinary.prediction_l1_relative'.format(data_source=data_source),
        "prediction_l2_relative" :  'results.best_network.algorithms.{data_source}.lsq.ordinary.prediction_l2_relative'.format(data_source=data_source),
        'wasserstein_speedup_raw' : "results.best_network.base_sampling_error.wasserstein_speedup",
        'wasserstein_speedup_real' : "results.best_network.base_sampling_error.wasserstein_speedup",
    }

    competitor_names = {
        "mean_error_relative" : sampling_method,
        "var_error_relative" : sampling_method,
        "wasserstein_error_cut" : sampling_method,
        "mean_bilevel_error_relative": sampling_method,
        "var_bilevel_error_relative" : sampling_method,
        "prediction_l1_relative": 'LSQ (with {})'.format(sampling_method),
        "prediction_l2_relative" : 'LSQ (with {})'.format(sampling_method),
         'wasserstein_speedup_raw' : sampling_method,
        'wasserstein_speedup_real' : sampling_method
    }

    extra_competitor_keys = {
        "mean_error_relative" : 'results.best_network.algorithms.{data_source}.lsq.ordinary.mean_error_relative'.format(data_source=data_source),
        "var_error_relative" : 'results.best_network.algorithms.{data_source}.lsq.ordinary.var_error_relative'.format(data_source=data_source),
        "wasserstein_error_cut" : 'results.best_network.algorithms.{data_source}.lsq.ordinary.wasserstein_error_cut'.format(data_source=data_source),
        "mean_bilevel_error_relative": 'results.best_network.algorithms.{data_source}.lsq.ordinary.mean_bilevel_error_relative'.format(data_source=data_source),
        "var_bilevel_error_relative" :'results.best_network.algorithms.{data_source}.lsq.ordinary.var_bilevel_error_relative'.format(data_source=data_source),
        'wasserstein_speedup_raw' : 'results.best_network.algorithms.{data_source}.lsq.ordinary.wasserstein_speedup_raw'.format(data_source=data_source),
        'wasserstein_speedup_real' : 'results.best_network.algorithms.{data_source}.lsq.ordinary.wasserstein_speedup_real'.format(data_source=data_source)
    }


    extra_competitor_names = {
        "mean_error_relative" : 'LSQ (with {})'.format(sampling_method),
        "var_error_relative" : 'LSQ (with {})'.format(sampling_method),
        "wasserstein_error_cut" : 'LSQ (with {})'.format(sampling_method),
        "mean_bilevel_error_relative": 'LSQ (with {})'.format(sampling_method),
        "var_bilevel_error_relative" : 'LSQ (with {})'.format(sampling_method),
         'wasserstein_speedup_raw' : 'LSQ (with {})'.format(sampling_method),
        'wasserstein_speedup_real' : 'LSQ (with {})'.format(sampling_method)
    }
    errors = {
    }



    errors_var = {}
    errors_min = {}

    errors_max = {}

    errors_retraining = {}
    errors_retraining_var = {}
    errors_retraining_min = {}
    errors_retraining_max = {}
    errors_min = {}

    errors_max = {}
    competitor = {}
    extra_competitors = {}
    for k in names.keys():
        errors[k] = np.zeros(len(train_sizes))
        errors_var[k]  = np.zeros(len(train_sizes))
        errors_max[k]  = np.zeros(len(train_sizes))
        errors_min[k]  = np.zeros(len(train_sizes))
        errors_retraining[k]  = np.zeros(len(train_sizes))
        errors_retraining_var[k]  = np.zeros(len(train_sizes))
        errors_retraining_max[k]  = np.zeros(len(train_sizes))
        errors_retraining_min[k]  = np.zeros(len(train_sizes))
        extra_competitors[k] = np.zeros(len(train_sizes))
        competitor[k]  = np.zeros(len(train_sizes))



    errors_per_network_size = {}
    for k in names.keys():
        errors_per_network_size[k] = []
        for i in range(len(train_sizes)):
            errors_per_network_size[k].append({})


    for error in errors.keys():

        tactics=['ordinary']#, 'replace']#, 'remove', 'add']
        error_per_tactics = {}
        var_error_per_tactics = {}
        min_error_per_tactics = {}
        max_error_per_tactics = {}
        error_per_tactics_retraining = {}
        var_error_per_tactics_retraining = {}
        min_error_per_tactics_retraining = {}
        max_error_per_tactics_retraining = {}


        pairing = {
            'mean_error': [error_per_tactics, errors],
            'var_error' : [var_error_per_tactics, errors_var],
            'min_error' : [min_error_per_tactics, errors_min],
            'max_error' : [max_error_per_tactics, errors_max],
            'mean_error_retraining' : [error_per_tactics_retraining, errors_retraining],
            'var_error_retraining' : [var_error_per_tactics_retraining, errors_retraining_var],
            'min_error_retraining' : [min_error_per_tactics_retraining, errors_retraining_min],
            'max_error_retraining' : [max_error_per_tactics_retraining, errors_retraining_max]
        }

        for t in tactics:
            for k in pairing.keys():
                pairing[k][0][t] = np.zeros(len(train_sizes))

        for t, tactic in enumerate(tactics):
            for (n, train_size) in enumerate(train_sizes):
                errors_local = []
                errors_local_regularization = {}
                errors_local_network_size = {}
                for configuration in data['configurations']:
                    ts = int(configuration['settings']['train_size'])
                    if ts == train_size:
                        errors_local.append(configuration['results']['best_network']['algorithms'][data_source]['ml'][tactic][error])
                        if not get_regularization_size(configuration) in errors_local_regularization.keys():
                            errors_local_regularization[get_regularization_size(configuration)] = []
                        errors_local_regularization[get_regularization_size(configuration)].append(configuration['results']['best_network']['algorithms'][data_source]['ml'][tactic][error])
                        competitor[error][n] = get_dict_path(configuration, competitor_keys[error])
                        has_extra_competitor = error in extra_competitor_keys.keys()
                        if error in extra_competitor_keys.keys():
                            extra_competitor = get_dict_path(configuration, extra_competitor_keys[error])
                            extra_competitors[error][n] = extra_competitor

                        for size_configuration in configuration['network_sizes']:
                            depth = get_dict_path(size_configuration, 'settings.depth')
                            width = get_dict_path(size_configuration, 'settings.max_width')

                            if depth not in errors_local_network_size.keys():
                                errors_local_network_size[depth] = {}
                            if width not in errors_local_network_size[depth].keys():
                                errors_local_network_size[depth][width] = []

                            error_at_size = size_configuration['results']['best_network']['algorithms'][data_source]['ml'][tactic][error]
                            if not np.isnan(error_at_size):
                                errors_local_network_size[depth][width].append(error_at_size)
                            else:
                                if 'speedup' in error:
                                    errors_local_network_size[depth][width].append(0.00001)
                                else:
                                    errors_local_network_size[depth][width].append(1)


                depths = np.array(sorted([k for k in errors_local_network_size.keys()]))
                widths = np.array(sorted([k for k in errors_local_network_size[depths[0]].keys()]))


                errors_per_width = []


                for width in widths:
                    errors_per_width.append([])
                    for depth in depths:
                        errors_per_width[-1].extend(errors_local_network_size[depth][width])




                errors_per_depth = []
                for depth in depths:
                    errors_per_depth.append([])
                    for width in widths:
                        errors_per_depth[-1].extend(errors_local_network_size[depth][width])




                errors_per_depth = np.array(errors_per_depth)
                errors_per_width = np.array(errors_per_width)


                if train_size == 128:
                    plt.figure()

                    plt.loglog(widths, np.mean(errors_per_width, axis=1), '-o', label='DNN selected retraining', basex=2, basey=2)
                    plt.loglog(widths, np.max(errors_per_width, axis=1), 'v', markersize=12, label='Max', basex=2)
                    plt.loglog(widths, np.min(errors_per_width, axis=1), '^', markersize=12, label='Min', basey=2)
                    plt.xlabel('Network width')
                    plt.ylabel(names[error])
                    plt.grid(True)
                    plt.legend()
                    plot_info.legendLeft()
                    if 'prediction' in error.lower():
                        plot_info.set_percentage_ticks(plt.gca().yaxis)
                    plt.title("{error} for {functional} as a function of width\nConfigurations: {title}\nUsing {train_size} samples\nTactic: {tactic}\nConfigurations per width: {configs_per_width}".format(error=names[error],
                                                                                                                                                              functional=functional, title=title, train_size=train_size, tactic=tactic,
                                                                                                                                                              configs_per_width = errors_per_width.shape[1]
                                                                                                                                                              ))
                    if only_network_sizes:
                        plot_info.savePlot("size_width_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                                                                                                        functional=functional, title=title, train_size=train_size, tactic=tactic
                                                                                                        ))


                    plt.close()

                    plt.figure()
                    plt.loglog(depths, np.mean(errors_per_depth, axis=1), '-o', label='DNN selected retraining', basex=2, basey=2)
                    plt.loglog(depths, np.max(errors_per_depth, axis=1), 'v', markersize=12, label='Max', basex=2, basey=2)
                    plt.loglog(depths, np.min(errors_per_depth, axis=1), '^', markersize=12, label='Min', basex=2, basey=2)
                    plt.xlabel('Network depth')
                    plt.ylabel(names[error])
                    if 'prediction' in error.lower():
                        plot_info.set_percentage_ticks(plt.gca().yaxis)
                    plt.grid(True)
                    plt.title("{error} for {functional} as a function of depth\nConfigurations: {title}\nUsing {train_size} samples\nTactic: {tactic}\nconfigs_per_depth = {configs_per_depth}".format(error=names[error],
                                                                                                                                                              functional=functional, title=title, train_size=train_size, tactic=tactic,
                                                                                                                                                              configs_per_depth=errors_per_depth.shape[1]
                    ))
                    plot_info.legendLeft()
                    if only_network_sizes:
                        plot_info.savePlot("size_depth_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                                                                                                           functional=functional, title=title, train_size=train_size, tactic=tactic
                                                                                                           ))

                    plt.close()

                plt.figure(10*(len(tactics)+1))
                plt.hist(errors_local, bins=20)
                if 'prediction' in error.lower():
                    plot_info.set_percentage_ticks(plt.gca().xaxis)
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples\nTactic: {tactic}\nNumber of configurations: {number_of_configurations}".format(error=names[error],
                    functional=functional, title=title, train_size=train_size,
                    tactic=tactic, number_of_configurations=len(errors_local)
                ))
                plot_info.savePlot("hist_no_competitor_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                    functional=functional, title=title, train_size=train_size, tactic=tactic
                ))


                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n], 0.5*np.diff(plt.gca().get_ylim())[0], competitor_names[error],rotation=90)
                if not only_network_sizes:
                    plot_info.savePlot("hist_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                        functional=functional, title=title, train_size=train_size, tactic=tactic
                    ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor,  0.5*np.diff(plt.gca().get_ylim())[0], extra_competitor_names[error],rotation=90)
                    if not only_network_sizes:
                        plot_info.savePlot("hist_lsq_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                        functional=functional, title=title, train_size=train_size,
                        tactic=tactic
                        ))

                plt.close(10*(len(tactics)+1))

                errors[error][n] = np.mean(errors_local)
                errors_var[error][n] = np.var(errors_local)
                errors_min[error][n] = np.amin(errors_local)
                errors_max[error][n] = np.amax(errors_local)



                errors_local_retrainings = []
                for configuration in data['configurations']:
                    ts = int(configuration['settings']['train_size'])
                    if ts == train_size:
                        retrainings = configuration['results']['retrainings'].keys()
                        for retraining in retrainings:
                            errors_local_retrainings.append(configuration['results']['retrainings'][retraining]['algorithms'][data_source]['ml'][tactic][error])
                errors_retraining[error][n] = np.mean(errors_local_retrainings)
                errors_retraining_var[error][n] = np.var(errors_local_retrainings)
                errors_retraining_min[error][n] = np.min(errors_local_retrainings)
                errors_retraining_max[error][n] = np.max(errors_local_retrainings)


                plt.figure(20*(len(tactics)+1))
                plt.hist(errors_local_retrainings, bins=20)
                if 'prediction' in error.lower():
                    plot_info.set_percentage_ticks(plt.gca().xaxis)
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution (retrainings) for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples\n Tactic: {tactic}".format(error=names[error],
                    functional=functional, title=title, train_size=train_size, tactic=tactic
                ))

                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n],  0.2*np.diff(plt.gca().get_ylim())[0], competitor_names[error],rotation=90)
                if not only_network_sizes:
                    plot_info.savePlot("hist_retraining_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                        functional=functional, title=title, train_size=train_size, tactic=tactic
                    ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor,  0.2*np.diff(plt.gca().get_ylim())[0], extra_competitor_names[error],rotation=90)
                    if not only_network_sizes:
                        plot_info.savePlot("hist_retraining_lsq_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                            functional=functional, title=title, train_size=train_size, tactic = tactic
                        ))

                plt.close(20*(len(tactics)+1))



                plt.figure(30*(len(tactics)+1))
                plt.hist(errors_local_retrainings, bins=20, alpha=0.5, label='Retrainings')
                plt.hist(errors_local, bins=20, alpha=0.5, label='Selected retrainings')
                if 'prediction' in error.lower():
                    plot_info.set_percentage_ticks(plt.gca().xaxis)
                plot_info.legendLeft()
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples\nTactic: {tactic}".format(error=names[error],
                    functional=functional, title=title, train_size=train_size, tactic=tactic
                ))

                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n], 0.2*np.diff(plt.gca().get_ylim())[0], competitor_names[error],rotation=90)

                if not only_network_sizes:
                    plot_info.savePlot("hist_both_retraining_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                        functional=functional, title=title, train_size=train_size, tactic=tactic
                    ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor, 0.2*np.diff(plt.gca().get_ylim())[0], extra_competitor_names[error],rotation=90)
                    if not only_network_sizes:
                        plot_info.savePlot("hist_both_retraining_lsq_{error}_{functional}_{title}_{train_size}_{tactic}".format(error=error,
                            functional=functional, title=title, train_size=train_size, tactic=tactic
                        ))

                plt.close(30*(len(tactics)+1))

                plt.figure(0)
                regularization_sizes = sorted([k for k in errors_local_regularization.keys()])
                reg_errors = [np.mean(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_var = [np.var(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_min = [np.min(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_max = [np.max(errors_local_regularization[k]) for k in regularization_sizes]
                #plt.errorbar(regularization_sizes, reg_errors, yerr=reg_errors_var, label=names[error], linewidth=3)
                plt.plot(regularization_sizes, reg_errors, '-o', label=names[error], linewidth=3)

                xticks_regularization_sizes = [0]
                xticks_regularization_sizes.append(regularization_sizes[-1]/2)
                xticks_regularization_sizes.append(regularization_sizes[-1])


                plt.xticks(xticks_regularization_sizes, ['{:.1e}'.format(k) for k in xticks_regularization_sizes])

                if 'speedup' not in error:
                    plt.plot(regularization_sizes, reg_errors_min, '--o', label='minimum', linewidth=3)
                else:
                    plt.plot(regularization_sizes, reg_errors_max, '--*', label='maximum', linewidth=3)
                plot_info.legendLeft()
                plt.title("Average error as a function of regularization size\n{functional}, configurations:{config},\nError: {error}, training size: {train_size}, tactic: {tactic}".
                    format(functional=functional, config=title, error=names[error], train_size=train_size, tactic=tactic))
                plt.xlabel("Regularization size")
                plt.ylabel(names[error])
                plt.gca().set_yscale("log", nonposy='clip', basey=2)
                if 'prediction' in error:
                    plot_info.set_percentage_ticks(plt.gca().yaxis)
                plt.grid(True)
                if not only_network_sizes:
                    plot_info.savePlot('error_regularization_{functional}_{config}_{error}_{train_size}_{tactic}'.format(tactic=tactic, functional=functional, config=title, error=error, train_size=train_size))
                plt.close('all')


                try:
                    plt.errorbar(regularization_sizes[1:], reg_errors[1:], yerr=reg_errors_var, label='error')
                    plt.plot(regularization_sizes[1:], reg_errors_min[1:], '--o', label='minimum')
                    plt.plot(regularization_sizes[1:], reg_errors_max[1:], '--*', label='maximum')
                    plot_info.legendLeft()
                    plt.title("Average error as a function of regularization size\n{functional}, configurations:{config},\nError: {error}, training size: {train_size}, tactic: {tactic}".
                        format(functional=functional, config=title, error=names[error], train_size=train_size, tactic=tactic))
                    plt.xlabel("Regularization size")
                    plt.ylabel(names[error])
                    plt.gca().set_yscale("log", nonposy='clip', basey=2)
                    plt.gca().set_xscale("log", nonposy='clip', basex=2)
                    plt.grid(True)
                    if not only_network_sizes:
                        plot_info.savePlot('error_regularization_log_{functional}_{config}_{error}_{train_size}_{tactic}'.format(tactic=tactic, functional=functional, config=title, error=error, train_size=train_size))
                    plt.close('all')
                except:
                    pass

                try:
                    plt.close('all')
                except:
                    pass


            for k in pairing.keys():
                pairing[k][0][tactic] = np.copy(pairing[k][1][error])

        if only_network_sizes:
            continue

        for tactic in tactics:
            table_builder = print_table.TableBuilder()
            header = ['Training size', 'DNN', 'DNN min', 'DNN max', 'DNN std', competitor_names[error]]
            if has_extra_competitor:
                header.append(extra_competitor_names[error])
            table_builder.set_header(header)
            for n, train_size in enumerate(train_sizes):
                row = [train_size, pairing['mean_error'][0][tactic][n], pairing['min_error'][0][tactic][n],
                    pairing['max_error'][0][tactic][n],
                    np.sqrt(pairing['var_error'][0][tactic][n]),
                    competitor[error][n]]

                if has_extra_competitor:
                    row.append(extra_competitors[error][n])

                table_builder.add_row(row)

            table_builder.set_title("{error} for selected retrained DNNs, with {functional}, configurations: {config}, tactic: {tactic}".format(
                    functional=functional,
                    config=title,
                    error = error,
                    tactic=tactic
            ))
            table_builder.print_table("selected_stats_{functional}_{config}_{error}_{tactic}".format(
                    functional=functional,
                    config=title,
                    error = error,
                    tactic=tactic
            ))

            table_builder = print_table.TableBuilder()
            header = ['Training size', 'DNN', 'DNN min', 'DNN max', 'DNN std', competitor_names[error]]
            if has_extra_competitor:
                header.append(extra_competitor_names[error])
            table_builder.set_header(header)
            for n, train_size in enumerate(train_sizes):
                row = [train_size, pairing['mean_error_retraining'][0][tactic][n], pairing['min_error'][0][tactic][n],
                    pairing['max_error_retraining'][0][tactic][n],
                    np.sqrt(pairing['var_error_retraining'][0][tactic][n]),
                    competitor[error][n]]

                if has_extra_competitor:
                    row.append(extra_competitors[error][n])

                table_builder.add_row(row)
            table_builder.set_title("{error} for all retrainings, with {functional}, configurations: {config}, tactic: {tactic}".format(
                            functional=functional,
                            config=title,
                            error = error,
                            tactic=tactic
            ))
            table_builder.print_table("retraining_tats_{functional}_{config}_{error}_{tactic}".format(
                    functional=functional,
                    config=title,
                    error = error,
                    tactic=tactic
            ))


            table_builder = print_table.TableBuilder()
            header = ['Training size', 'DNN', competitor_names[error]]
            if has_extra_competitor:
                header.append(extra_competitor_names[error])
            table_builder.set_header(header)
            for n, train_size in enumerate(train_sizes):
                row = [train_size, pairing['mean_error_retraining'][0][tactic][n],
                    competitor[error][n]]

                if has_extra_competitor:
                    row.append(extra_competitors[error][n])

                table_builder.add_row(row)

            table_builder.set_title("{error} for selected retrained DNNs, with {functional}, configurations: {config}, tactic: {tactic}".format(
                        functional=functional,
                        config=title,
                        error = error,
                        tactic=tactic
            ))
            table_builder.print_table("selected_{functional}_{config}_{error}_{tactic}".format(
                    functional=functional,
                    config=title,
                    error = error,
                    tactic=tactic
            ))






        on_off_array = [True, False]
        off_array = [False]
        on_array = [True]
        for include_selected in on_array:
            for include_retraining in off_array:
                for include_min in on_off_array:
                    for include_max in on_off_array:
                        for include_std in off_array:
                            for include_competitor in on_off_array:
                                for tactics_in_same_plot in off_array:
                                    for include_extra_competitor in [include_competitor]:
                                        for tactic in tactics:

                                            if tactics_in_same_plot:
                                                tactic_added_name = " ({tactic})".format(tactic=tactic)
                                            else:
                                                tactic_added_name = ""
                                            if include_selected:
                                                if include_std:
                                                    p = plt.errorbar(train_sizes, pairing['mean_error'][0][tactic],
                                                        yerr=np.sqrt(pairing['mean_error'][0][tactic]),
                                                        label='DNN selected retraining', ls='--',
                                                        solid_capstyle='projecting', capsize=5, linewidth =3)
                                                else:
                                                    p = plt.loglog(train_sizes, pairing['mean_error'][0][tactic], '--*',
                                                        label='DNN selected retraining' + tactic_added_name, basex=2, basey=2, linewidth=3, markersize=15)
                                                if 'speedup' not in error:
                                                    poly = np.polyfit(np.log(train_sizes), np.log(pairing['mean_error'][0][tactic]), 1)

                                                    constant = np.exp(poly[1])
                                                    if constant <= 10 and constant >= 1:
                                                        constant_latex = str(constant)
                                                    else:
                                                        power = int(np.log10(constant))

                                                        r = constant / 10**(power-1)

                                                        constant_latex = "{:.2f}\\cdot 10^{{{}}}".format(r, power-1)

                                                    plt.loglog(train_sizes, np.exp(poly[1])*train_sizes**poly[0],
                                                               '--', label='$%s\\cdot N^{%.2f}$' % (constant_latex, poly[0]))

                                                if include_max:
                                                    plt.loglog(train_sizes, pairing['max_error'][0][tactic], 'v', label='Max DNN selected retraining' + tactic_added_name,
                                                                color=p[0].get_color(),
                                                                markersize = 12)
                                                if include_min:
                                                    plt.loglog(train_sizes, pairing['min_error'][0][tactic], '^', label='Min DNN selected retraining' + tactic_added_name,
                                                        markersize=12,
                                                                color=p[0].get_color())

                                            if include_retraining:
                                                 if include_std:
                                                     p = plt.errorbar(train_sizes, pairing['mean_error_retraining'][0][tactic],
                                                         yerr=np.sqrt(pairing['var_error_retraining'][0][tactic]),
                                                         label='DNN all Retrainings' + tactic_added_name,
                                                         solid_capstyle='projecting', capsize=5)
                                                 else:
                                                     p = plt.loglog(train_sizes, pairing['mean_error_retraining'][0][tactic], '-o',
                                                         label='DNN all retrainings', basex=2, basey=2)

                                                 if include_max:
                                                     plt.loglog(train_sizes, pairing['max_error_retraining'][0][tactic], '.', label='Max DNN all retrainings' + tactic_added_name,
                                                                 color=p[0].get_color())
                                                 if include_min:
                                                     plt.loglog(train_sizes, pairing['min_error'][0][tactic], 'x', label='Min DNN all retrainings' + tactic_added_name,
                                                                 color=p[0].get_color())

                                            if include_competitor and not tactics_in_same_plot:
                                                plt.loglog(train_sizes, competitor[error], '--o', label=competitor_names[error], basex=2, basey=2)
                                            if include_extra_competitor and not tactics_in_same_plot and has_extra_competitor:
                                                plt.loglog(train_sizes, extra_competitors[error], '--P', label=extra_competitor_names[error], basex=2, basey=2)
                                            plt.gca().set_xscale("log", nonposx='clip', basex=2)
                                            plt.gca().set_yscale("log", nonposy='clip', basey=2)
                                            plt.grid(True)
                                            plt.xlabel("Number of traning samples ($N$)")
                                            plt.ylabel(names[error])
                                            if 'prediction' in error:
                                                plot_info.set_percentage_ticks(plt.gca().yaxis)
                                            if not tactics_in_same_plot:
                                                plot_info.legendLeft()
                                                plt.title("{error} as a function of training samples\n({functional}, {tactic})\nConfigurations: {config}".\
                                                    format(error=names[error], tactic=tactic, functional=functional, config=title))

                                                if not only_network_sizes:
                                                    plot_info.savePlot(generate_plot_name(error, functional, tactic, title, include_selected,
                                                                    include_retraining, include_min, include_max, include_std, include_competitor,
                                                                    include_extra_competitor and has_extra_competitor,
                                                                    tactics_in_same_plot))

                                                if tactic == 'ordinary' and include_competitor and include_min and include_max:
                                                    plt.show()
                                                plt.close('all')





                                        if tactics_in_same_plot:
                                            if include_competitor:
                                                plt.loglog(train_sizes, competitor[error], '--o', label=competitor_names[error], basex=2, basey=2)

                                            if include_extra_competitor and has_extra_competitor:
                                                plt.loglog(train_sizes, extra_competitors[error], '--P', label=extra_competitor_names[error], basex=2, basey=2)

                                            plot_info.legendLeft()
                                            plt.title("{error} as a function of training samples\n({functional})\nConfigurations: {config}".\
                                                format(error=names[error], functional=functional, config=title))

                                            if not only_network_sizes:
                                                plot_info.savePlot(generate_plot_name(error, functional, "All tactics", title, include_selected,
                                                                include_retraining, include_min, include_max, include_std, include_competitor,
                                                                include_extra_competitor and has_extra_competitor,
                                                                tactics_in_same_plot))



                                            plt.close('all')




def filter_configs(data, excludes={}, onlys={}, test_functions = []):
    data_copy = {}

    for k in data.keys():
        if k != 'configurations':
            data_copy[k] = copy.deepcopy(data[k])
    data_copy['configurations'] = []

    for config in data['configurations']:
        keep = True
        for exclude_path in excludes.keys():
            split = exclude[0].split('.')
            value = config
            for k in split:
                value = value[k]
            excluded_values = excludes[exclude_path]

            for excluded_value in excluded_values:
                if value == excluded_value:
                    keep = False
        if not keep:
            continue

        for only_path in onlys.keys():
            split = only_path.split('.')
            value = config
            for k in split:
                value = value[k]
            only_values = onlys[only_path]

            equal_to_one = False
            for only_value in only_values:
                if value == only_value:
                    equal_to_one = True
            if not equal_to_one:
                keep = False

        if not keep:
            continue


        for test_function in test_functions:
            if not test_function(config):
                keep = False

        if not keep:
            continue


        data_copy['configurations'].append(copy.deepcopy(config))

    return data_copy


# In[8]:

def get_selection(config):
    return config['settings']['selction']

def has_regularization(config):
    return config['settings']['regularizer'] is not None and config['settings']['regularizer'] != "None"

def get_regularization(config):
    return config['settings']['regularizer']

def get_optimizer(config):
    return config['settings']['optimizer']

def get_loss(config):
    return config['settings']['loss']

def only_adam_and_no_regularization_for_mse(config):
    return get_optimizer(config) == 'Adam' and (get_loss(config) == 'mean_absolute_error' or not has_regularization(config))

def only_adam_and_no_regularization_for_mse_and_reg_for_l1(config):
    if get_optimizer(config) != 'Adam':
        return False
    if get_loss(config) == 'mean_absolute_error':
        return has_regularization(config)
    else:
        return not has_regularization(config)

def only_adam_and_low_regularization_for_mse_and_reg_for_l1(config):
    if get_optimizer(config) != 'Adam':
        return False
    if get_loss(config) == 'mean_absolute_error':
        return has_regularization(config)
    else:
        return (not has_regularization(config)) or \
            (get_regularization_type(config) == "l2" \
                and get_regularization_size(config) < 1e-5)

def get_only_sgd(config):
    return get_optimizer(config) == "SGD"

def get_only_adam(config):
    return get_optimizer(config) == "Adam"

def get_only_mse(config):
    return get_loss(config) == 'mean_squared_error'

def get_only_mean_m2(config):
    return get_loss(config) == "mean_m2"

def get_only_mae(config):
    return get_loss(config) == 'mean_absolute_error'

def complement(f):
    return lambda x: not f(x)

def and_config(f1, f2):
    return lambda x: f1(x) and f2(x)

def only_l1_reg(config):
    return get_regularization_type(config) == "l1"

def only_l2_reg(config):
    return get_regularization_type(config) == "l2"

def get_regularization_type(config):
    if not has_regularization(config):
        return "None"
    reg = get_regularization(config)

    if reg['l1'] > 0:
        return "l1"
    else:
        return "l2"

def only_wasserstein_train(config):
    return get_selection(config) == 'wasserstein_train'

def only_ray_prediction(config):
    return get_selection(config) == 'ray_prediction'

def only_train(config):
    return get_selection(config) == 'train'

def only_mean_train(config):
    return get_selection(config) == 'mean_train'



def compare_two_sets(functional, *, data1, title1, data2, title2, main_title):
    if len(data1['configurations']) + len(data2['configurations']) == 0:

        print("No configurations!")

        return

    train_sizes = []

    for configuration in data1['configurations']:
        train_size = int(configuration['settings']['train_size'])
        if train_size not in train_sizes:
            train_sizes.append(train_size)

    train_sizes = sorted(train_sizes)

    data_source_names = data1['configurations'][0]['results']['best_network']['algorithms'].keys()
    for k in data_source_names:
        if re.match(r'^[Q]?MC_from_data$', k):
            data_source = k
            break

    sampling_method = re.search(r'(^[Q]?MC)_from_data$', data_source).group(1)
    names = {
        #"mean_error_relative" : "mean relative error",
        #"var_error_relative" : "variance relative error",
        "wasserstein_error_cut" : "Wasserstein",
        #"mean_bilevel_error_relative": "relative error bilevel mean",
        #"var_bilevel_error_relative" :"relative error bilevel variance",
        #"prediction_l1_relative": 'relative prediction error ($L^1$)',
        "prediction_l2_relative" : 'relative prediction error ($L^2$)',
        'wasserstein_speedup_raw' : 'Raw Wasserstein speedup',
        'wasserstein_speedup_real' : 'Wasserstein speedup with convergence rate',
    }


    for error in names.keys():

        tactics=['ordinary']#, 'replace']#, 'remove', 'add']






        sources = {title1: data1, title2: data2}
        for t, tactic in enumerate(tactics):
            for (n, train_size) in enumerate(train_sizes):
                if train_size != 128:
                    continue
                errors_local = {}
                for source in sources.keys():
                    errors_local[source] = []
                for source_name in sources.keys():
                    for configuration in sources[source_name]['configurations']:
                        ts = int(configuration['settings']['train_size'])
                        if ts == train_size:

                            errors_local[source_name].append(configuration['results']['best_network']['algorithms'][data_source]['ml'][tactic][error])

                source_names = [k for k in sources.keys()]

                min_value = np.amin([np.amin(errors_local[source]) for source in source_names])
                max_value = np.amax([np.amax(errors_local[source]) for source in source_names])
                number_in_each_str = ""
                for source in sources.keys():
                    number_in_each_str += "{}: {} configurations\n".format(source, len(errors_local[source]))
                    plt.hist(errors_local[source], bins=20, label=source, alpha=0.5, range=[min_value, max_value])

                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")

                plt.legend()
                if 'prediction' in error.lower():
                    plot_info.set_percentage_ticks(plt.gca().xaxis)
                plt.title("Comparison histograms for distribution for {error} for {functional}\nConfigurations: {title1} versus {title2}\n{main_title}\nUsing {train_size} samples\nTactic: {tactic}\nNumber of configurations:\n{number_in_each_str}".format(
                    error=names[error],
                    functional=functional, title2=title2, title1=title1, train_size=train_size,
                    main_title = main_title,
                    tactic=tactic,
                    number_in_each_str = number_in_each_str
                ))

                plot_info.savePlot('comparison_histogram_{error}_{functional}_{title1}_{title2}_{train_size}_{tactic}_{main_title}'.format(
                    functional=functional, title1 = title1, title2 = title2, train_size = train_size, tactic = tactic, error = error, main_title=main_title
                ))

                #if train_size == 128:
                #   plt.show()
                plt.close('all')
