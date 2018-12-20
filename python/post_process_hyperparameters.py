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


def fix_bilevel(configuration):
    """
    In an earlier verison (4f65635f5b32b842b8b5c80f9978520c85545b25) there was an error in how
    the error (yes) of bilevel was calculated. This fixes this, and adds relative errors
    """

    for config in configuration['configurations']:
        sources = [config['results']['best_network']['algorithms']]
        sources.extend([config['results']['retrainings'][k]['algorithms'] for k in config['results']['retrainings'].keys()])
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










def plot_all(filenames, convergence_rate, latex_out):
    functionals = [k for k in filenames.keys()]
    data = {}

    for functional in functionals:
        data[functional] = []

        filename =filenames[functional]

        with open(filename, 'r') as f:
            json_content = json.load(f)
            fix_bilevel(json_content)
            add_wasserstein_speedup(json_content, convergence_rate)
            data[functional] = copy.deepcopy(json_content)


    onlys = {
        "" : {},
        "Best performing": {"settings.selection_type":["Best performing"]},
        "Emperically optimal" :{"settings.selection_type":["Emperically optimal"]},
    }


    filters = {
        'All configurations' : lambda x: True,
        'Only Adam with ($L^2$ and no regularization) or ($L^1$)' : only_adam_and_no_regularization_for_mse,
        'Only Adam with ($L^2$ and no regularization) or ($L^1$ with regularization)' : only_adam_and_no_regularization_for_mse_and_reg_for_l1,
    }

    for f in [best_configuration_1, best_configuration_2, best_configuration_3, best_configuration_4]:
        filters[f.__doc__] = f


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

    for functional in functionals:
        heading(0, functional)
        for filtername in filters:
            heading(1, filtername)
            for only in onlys:
                heading(2, only)
                plot_as_training_size(functional, filter_configs(data[functional], test_functions=[filters[filtername]],
                                    onlys=onlys[only]), \
                                  filtername+" " + only)

    with open(latex_out) as f:
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
        self.text +=         """
%%%%%%%%%%%%%
% {full_path}
\\begin{{figure}}
\\InputImage{{0.8\\textwidth}}{{0.6\\textheight}}{{{image_name}}}
\\cprotect\\caption{{{title}\\\\
\\textbf{{To include:}}\\\\ \\verb|\\InputImage{{0.8\\textwidth}}{{0.6\\textheight}}{{{image_name}}}|\\\\
Full path:\\
(\\verb|{full_path}|)
}}

\\end{{figure}}
""".format(image_name=basename, full_path=image_path, title=title)

    def get_latex(self):
        return self.text + "\n\\end{document}"


    def add_table(self, tablefile, title):
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
def plot_as_training_size(functional, data, title="all configurations"):
    train_sizes = []

    for configuration in data['configurations']:
        train_size = int(configuration['settings']['train_size'])
        if train_size not in train_sizes:
            train_sizes.append(train_size)
    print(train_sizes)
    train_sizes = sorted(train_sizes)

    data_source_names = data['configurations'][0]['results']['best_network']['algorithms'].keys()
    for k in data_source_names:
        if re.match(r'^[Q]?MC_from_data$', k):
            data_source = k
            break

    sampling_method = re.search(r'(^[Q]?MC)_from_data$', data_source).group(1)
    names = {
        "mean_error_relative" : "mean relative error",
        "var_error_relative" : "variance relative error",
        "wasserstein_error_cut" : "Wasserstein",
        "mean_bilevel_error_relative": "relative error bilevel mean",
        "var_bilevel_error_relative" :"relative error bilevel variance",
        "prediction_l1_relative": 'relative prediction error ($L^1$)',
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


    for error in errors.keys():

        tactics=['ordinary', 'add', 'remove', 'replace']
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
                plt.figure(10*(len(tactics)+1))
                plt.hist(errors_local, bins=20)
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples".format(error=names[error],
                    functional=functional, title=title, train_size=train_size
                ))

                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n],2, competitor_names[error],rotation=90)
                plot_info.savePlot("hist_{error}_{functional}_{title}_{train_size}".format(error=error,
                    functional=functional, title=title, train_size=train_size
                ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor, 2, extra_competitor_names[error],rotation=90)
                    plot_info.savePlot("hist_lsq_{error}_{functional}_{title}_{train_size}".format(error=error,
                        functional=functional, title=title, train_size=train_size
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
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution (retrainings) for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples".format(error=names[error],
                    functional=functional, title=title, train_size=train_size
                ))

                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n],2, competitor_names[error],rotation=90)
                plot_info.savePlot("hist_retraining_{error}_{functional}_{title}_{train_size}".format(error=error,
                    functional=functional, title=title, train_size=train_size
                ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor,2, extra_competitor_names[error],rotation=90)
                    plot_info.savePlot("hist_retraining_lsq_{error}_{functional}_{title}_{train_size}".format(error=error,
                        functional=functional, title=title, train_size=train_size
                    ))

                plt.close(20*(len(tactics)+1))



                plt.figure(30*(len(tactics)+1))
                plt.hist(errors_local_retrainings, bins=20, alpha=0.5, label='Retrainings')
                plt.hist(errors_local, bins=20, alpha=0.5, label='Selected retrainings')
                plot_info.legendLeft()
                plt.xlabel(names[error])
                plt.ylabel("Number of configurations")
                plt.title("Histograms for distribution for {error} for {functional}\nConfigurations: {title}\nUsing {train_size} samples".format(error=names[error],
                    functional=functional, title=title, train_size=train_size
                ))

                plt.axvline(x=competitor[error][n], linestyle='--',color='grey')
                plt.text(competitor[error][n],2, competitor_names[error],rotation=90)
                plot_info.savePlot("hist_both_retraining_{error}_{functional}_{title}_{train_size}".format(error=error,
                    functional=functional, title=title, train_size=train_size
                ))

                if error in extra_competitor_keys.keys():
                    plt.axvline(x=extra_competitor, linestyle='--',color='green')
                    plt.text(extra_competitor,2, extra_competitor_names[error],rotation=90)
                    plot_info.savePlot("hist_both_retraining_lsq_{error}_{functional}_{title}_{train_size}".format(error=error,
                        functional=functional, title=title, train_size=train_size
                    ))

                plt.close(30*(len(tactics)+1))

                plt.figure(0)
                regularization_sizes = sorted([k for k in errors_local_regularization.keys()])
                reg_errors = [np.mean(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_var = [np.var(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_min = [np.min(errors_local_regularization[k]) for k in regularization_sizes]
                reg_errors_max = [np.max(errors_local_regularization[k]) for k in regularization_sizes]
                plt.errorbar(regularization_sizes, reg_errors, yerr=reg_errors_var, label='error')
                plt.plot(regularization_sizes, reg_errors_min, '--o', label='minimum')
                plt.plot(regularization_sizes, reg_errors_max, '--*', label='maximum')
                plot_info.legendLeft()
                plt.title("Average error as a function of regularization size\n{functional}, configurations:{config},\nError: {error}, training size: {train_size}, tactic: {tactic}".
                    format(functional=functional, config=title, error=names[error], train_size=train_size, tactic=tactic))
                plt.xlabel("Regularization size")
                plt.ylabel(names[error])
                plt.gca().set_yscale("log", nonposy='clip', basey=2)
                plt.grid(True)
                plot_info.savePlot('error_regularization_{functional}_{config}_{error}_{train_size}_{tactic}'.format(tactic=tactic, functional=functional, config=title, error=error, train_size=train_size))
                plt.close('all')


            for k in pairing.keys():
                pairing[k][0][tactic] = np.copy(pairing[k][1][error])


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

        for include_selected in on_off_array:
            for include_retraining in on_off_array:
                for include_min in on_off_array:
                    for include_max in on_off_array:
                        for include_std in on_off_array:
                            for include_competitor in on_off_array:
                                for tactics_in_same_plot in on_off_array:
                                    for include_extra_competitor in on_off_array:
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
                                                        solid_capstyle='projecting', capsize=5)
                                                else:
                                                    p = plt.loglog(train_sizes, pairing['mean_error'][0][tactic], '--*',
                                                        label='DNN selected retraining' + tactic_added_name, basex=2, basey=2)

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
                                            plt.xlabel("Number of traning samples")
                                            plt.ylabel(names[error])
                                            if not tactics_in_same_plot:
                                                plot_info.legendLeft()
                                                plt.title("{error} as a function of training samples\n({functional}, {tactic})\nConfigurations: {config}".\
                                                    format(error=names[error], tactic=tactic, functional=functional, config=title))
                                                plot_info.savePlot(generate_plot_name(error, functional, tactic, title, include_selected,
                                                                include_retraining, include_min, include_max, include_std, include_competitor,
                                                                include_extra_competitor and has_extra_competitor,
                                                                tactics_in_same_plot))
                                                plt.close('all')





                                        if tactics_in_same_plot:
                                            if include_competitor:
                                                plt.loglog(train_sizes, competitor[error], '--o', label=competitor_names[error], basex=2, basey=2)

                                            if include_extra_competitor and has_extra_competitor:
                                                plt.loglog(train_sizes, extra_competitors[error], '--P', label=extra_competitor_names[error], basex=2, basey=2)

                                            plot_info.legendLeft()
                                            plt.title("{error} as a function of training samples\n({functional})\nConfigurations: {config}".\
                                                format(error=names[error], functional=functional, config=title))
                                            plot_info.savePlot(generate_plot_name(error, functional, "All tactics", title, include_selected,
                                                            include_retraining, include_min, include_max, include_std, include_competitor,
                                                            include_extra_competitor and has_extra_competitor,
                                                            tactics_in_same_plot))

                                            if include_competitor and include_extra_competitor and include_selected and include_max and include_min and include_retraining and not include_std:
                                                plt.show()

                                            plt.close('all')




def filter_configs(data, excludes={}, onlys={}, test_functions = []):
    data_copy = {}

    for k in data.keys():
        if k != 'configurations':
            data_copy[k] = copy.deepcopy(data[k])
    data_copy['configurations'] = []
    print(len(data['configurations']))
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
    print(len(data_copy['configurations']))
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

def best_configuration_1(config):
    """Adam, mean_absolute_error, ray_prediction, l2 reg"""
    if get_optimizer(config) != 'Adam':
        return False

    if get_loss(config) != 'mean_absolute_error':
        return False

    if get_selection(config) != 'ray_prediction':
        return False

    if not has_regularization(config):
        return False

    return get_regularization(config)['l2'] > 0 and get_regularization(config)['l1'] == 0

def best_configuration_2(config):
    """Adam, mean_squared_error, train, No reg"""
    if get_optimizer(config) != 'Adam':
        return False

    if get_loss(config) != 'mean_squared_error':
        return False

    if get_selection(config) != 'train':
        return False

    return not has_regularization(config)


def best_configuration_3(config):
    """Adam, mean_absolute_error, wasserstein_train, l2 reg"""
    if get_optimizer(config) != 'Adam':
        return False

    if get_loss(config) != 'mean_absolute_error':
        return False

    if get_selection(config) != 'wasserstein_train':
        return False

    if not has_regularization(config):
        return False

    return get_regularization(config)['l2'] > 0 and get_regularization(config)['l1'] == 0


def best_configuration_4(config):
    """Adam, mean_absolute_error, ray_prediction, l1 reg"""
    if get_optimizer(config) != 'Adam':
        return False

    if get_loss(config) != 'mean_absolute_error':
        return False

    if get_selection(config) != 'ray_prediction':
        return False

    if not has_regularization(config):
        return False

    return get_regularization(config)['l1'] > 0 and get_regularization(config)['l2'] == 0


# In[ ]:





# In[ ]: