import glob
import os
import sys
import json
import copy
import re
import git
import datetime
import numpy as np
basename = sys.argv[1]
functional_name = sys.argv[2]
outname = sys.argv[3]

folders = glob.glob('{}_*'.format(basename))
configuration_top = {}
configuration_top['functional_name'] = functional_name
git_repo = git.Repo(path=folders[0], search_parent_directories=True)

configuration_top['data_git_version'] = git_repo.head.object.hexsha
configuration_top['data_git_url'] = git_repo.remotes.origin.url

script_git_repo = git.Repo(search_parent_directories=True)

configuration_top['script_git_version'] = script_git_repo.head.object.hexsha
configuration_top['script_git_url'] = script_git_repo.remotes.origin.url

configuration_top['configurations'] = []
configuration_top['command_issued'] = "{} {}".format(sys.executable, " ".join(sys.argv))
configuration_top['generated_at'] = datetime.datetime.now().isoformat()
configuration_top['working_directory'] = os.getcwd()


# The following two lines limits the output severely.
only_data_sources = ['MC_from_data']
only_tactics = ['ordinary']
only_competitor_retraining_network_size = ['ml']
only_keys_retraining_network_size = ['algorithms', 'mc_errors']


def filter_config(config, only_tactics, only_data_sources, only_competitor_retraining_network_size, only_keys_retraining_network_size):
    """
    Filters out unused information from the configuration (otherwise each output file easily gets up to 18 GB)
    """


    sub_configs = [config['results']['best_network'],
                   *[config['network_sizes'][k]['results']['best_network'] for k in range(len(config['network_sizes']))],
                   *[config['results']['retrainings'][k] for k in config['results']['retrainings'].keys()]]

    for k in range(len(config['network_sizes'])):
        for n in config['network_sizes'][k]['results']['retrainings'].keys():
            sub_configs.append(config['network_sizes'][k]['results']['retrainings'][n])
    
    for sub_config in sub_configs:
        # filter out data sources
        data_sources_to_delete = []
        for data_source_name in sub_config['algorithms'].keys():
            keep = False

            for allowed_data_source in only_data_sources:
                if allowed_data_source in data_source_name:
                    keep = True

            if not keep:
                data_sources_to_delete.append(data_source_name)
        for data_source_name in data_sources_to_delete:
            del sub_config['algorithms'][data_source_name]

        # filter out tacticsn:
        for data_source_name in sub_config['algorithms'].keys():
            for competitor in sub_config['algorithms'][data_source_name].keys():
                tactics_to_delete = []
                for tactic in sub_config['algorithms'][data_source_name][competitor].keys():
                    keep = False
                    for allowed_tactic in only_tactics:
                        if allowed_tactic == tactic:
                            keep = True

                    if not keep:
                        tactics_to_delete.append(tactic)
                for tactic in tactics_to_delete:
                    del sub_config['algorithms'][data_source_name][competitor][tactic]


    # First filter out non used  keys:
    configs_size_retrainings = [*[config['network_sizes'][k]['results']['best_network'] for k in range(len(config['network_sizes']))],
                                *[config['results']['retrainings'][k] for k in config['results']['retrainings'].keys()]]

    for k in range(len(config['network_sizes'])):
        for n in config['network_sizes'][k]['results']['retrainings'].keys():
            configs_size_retrainings.append(config['network_sizes'][k]['results']['retrainings'][n])



    for config_top in configs_size_retrainings:
        remove_top_level = []
        for field in config_top.keys():
            keep = False

            for allowed_field in only_keys_retraining_network_size:
                if allowed_field == field:
                    keep = True

            if not keep:
                remove_top_level.append(field)

        for field in remove_top_level:
            del config_top[field]
    
    


    # filter out other algorithms for the non-main configuration
    sub_configs_size_retrainings = [*[config['network_sizes'][k]['results']['best_network'] for k in range(len(config['network_sizes']))],
                                    *[config['results']['retrainings'][k] for k in config['results']['retrainings'].keys()]]


    for k in range(len(config['network_sizes'])):
        for n in config['network_sizes'][k]['results']['retrainings'].keys():
            configs_size_retrainings.append(config['network_sizes'][k]['results']['retrainings'][n])

    
    

    for sub_config in sub_configs_size_retrainings:
        for data_source_name in sub_config['algorithms'].keys():
            competitors_to_delete = []
            for competitor in sub_config['algorithms'][data_source_name].keys():
                keep = False
                for allowed_competitor in only_competitor_retraining_network_size:
                    if allowed_competitor == competitor:
                        keep = True
                if not keep:
                    competitors_to_delete.append(competitor)
            for competitor in competitors_to_delete:
                del sub_config['algorithms'][data_source_name][competitor]


for folder in folders:
    lsf = glob.glob(os.path.join(folder, 'lsf.o*'))
    if len (lsf) == 0:
        continue
    lsf_name = lsf[0]
    with open(lsf_name) as lsf_file:
        lsf_content = lsf_file.read()
    
    if 'Successfully completed.' not in lsf_content:
        continue
    
    config_file_name = os.path.join(folder, 'config_run.json')

    
    with open(config_file_name) as config_file:
        json_config = json.load(config_file)
    best_files = glob.glob(os.path.join(folder, 'results/*{}*combination_stats.json'.format(functional_name)))
    configuration = {}

    configuration['results'] = {}
    configuration['network_sizes'] = []
    configuration['settings'] = copy.deepcopy(json_config)
    configuration['settings']['from_folder'] = folder
    for best_file in best_files:
        with open(best_file) as best_result_file:
            loaded_json = json.load(best_result_file)
            if 'network_size' not in best_file:
                configuration['results']['best_network'] = copy.deepcopy(loaded_json)

                current_configuration = configuration
            else:
                configuration['network_sizes'].append({})
                current_configuration = configuration['network_sizes'][-1]
                current_configuration['results'] = {}
                current_configuration['settings'] = copy.deepcopy(json_config)
                current_configuration['results']['best_network'] = copy.deepcopy(loaded_json)
    
        
    
        basename = re.search(r'results\/(.+)_combination_stats\.json', best_file).group(1)
        
        model_name = '{}model.json'.format(basename)
        with open(os.path.join(os.path.join(folder, 'results'),model_name)) as model_file:
            model = json.load(model_file)
            depth = len(model['config']['layers'])
            widths = [model['config']['layers'][k]['config']['units'] for k in range(depth)]
            
        current_configuration['settings']['depth'] = copy.deepcopy(depth)
        current_configuration['settings']['widths'] = copy.deepcopy(widths)
        current_configuration['settings']['min_width'] = min(widths)
        current_configuration['settings']['max_width'] = max(widths)
        current_configuration['settings']['average_width'] = np.mean(widths)
        
        
                
        
        current_configuration['results']['retrainings'] = {}
    
        retraining_files = glob.glob(os.path.join(folder, 'results/{basename}_combination_stats_try_*.json'.format(basename=basename)))
        for retraining_file_name in retraining_files:
            retraining_number = int(re.search(r'combination_stats_try_(\d+)\.json', retraining_file_name).group(1))
            with open(retraining_file_name) as retraining_file:
                loaded_retraining_json = json.load(retraining_file)

                current_configuration['results']['retrainings'][retraining_number] = copy.deepcopy(loaded_retraining_json)
    
    

    filter_config(configuration, only_tactics, only_data_sources, only_competitor_retraining_network_size, only_keys_retraining_network_size)
                                
    configuration_top['configurations'].append(configuration)

import pickle
import bz2
import gzip
compressors = {"bz2":  bz2.BZ2File,
               "gz" : gzip.GzipFile}
try:
    with open('{}.pic'.format(outname), 'w') as out:
        pickle.dump(configuration_top, out)
except:
    pass

with open('{}.json'.format(outname), 'w') as out:
    json.dump(configuration_top, out)

for compressor in compressors:
    # see https://stackoverflow.com/a/39451012
    json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)
        
    with compressors[compressor]('{}.json.{}'.format(outname, compressor), 'w') as out:
        out.write(json_bytes)
        
    
                                 
