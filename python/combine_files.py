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
    for best_file in best_files:
        configuration = {}
        configuration['results'] = {}
        with open(best_file) as best_result_file:
            configuration['results']['best_network'] = copy.deepcopy(json.load(best_result_file))
    
        
        configuration['settings'] = copy.deepcopy(json_config)
    
        basename = re.search(r'results\/(.+)_combination_stats\.json', best_file).group(1)
        
        model_name = '{}model.json'.format(basename)
        with open(os.path.join(os.path.join(folder, 'results'),model_name)) as model_file:
            model = json.load(model_file)
            depth = len(model['config']['layers'])
            widths = [model['config']['layers'][k]['config']['units'] for k in range(depth)]
            
        configuration['settings']['depth'] = copy.deepcopy(depth)
        configuration['settings']['widths'] = copy.deepcopy(widths)
        configuration['settings']['min_width'] = min(widths)
        configuration['settings']['max_width'] = max(widths)
        configuration['settings']['average_width'] = np.mean(widths)
        
        
                
        
        configuration['results']['retrainings'] = {}
    
        retraining_files = glob.glob(os.path.join(folder, 'results/{basename}_combination_stats_try_*.json'.format(basename=basename)))
        for retraining_file_name in retraining_files:
            retraining_number = int(re.search(r'combination_stats_try_(\d+)\.json', retraining_file_name).group(1))
            with open(retraining_file_name) as retraining_file:
                configuration['results']['retrainings'][retraining_number] = json.load(retraining_file)
    
    

    
                                
        configuration_top['configurations'].append(configuration)


print(json.dumps(configuration_top))

    
                                 
