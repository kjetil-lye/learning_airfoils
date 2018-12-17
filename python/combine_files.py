import glob
import os
import sys
import json
import copy
import re
import git
import datetime
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
configuration_top['command_issued'] = "{} {}".format(sys.executable + " ".join(sys.argv))
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
    configuration = {}
    with open(config_file_name) as config_file:
        json_config = json.load(config_file)
    configuration['settings'] = copy.deepcopy(json_config)
    
    configuration['results'] = {}
    configuration['results']['retrainings'] = {}
    
    retraining_files = glob.glob(os.path.join(folder, 'results/*{}*combination_stats_try_*.json'.format(functional_name)))
    for retraining_file_name in retraining_files:
        if 'network_size' in retraining_file_name:
            continue
        retraining_number = int(re.search(r'combination_stats_try_(\d+)\.json', retraining_file_name).group(1))
        with open(retraining_file_name) as retraining_file:
            configuration['results']['retrainings'][retraining_number] = json.load(retraining_file)
    
    best_file = glob.glob(os.path.join(folder, 'results/*{}*combination_stats.json'.format(functional_name)))

    for best_file_name in best_file:
        if 'network_size' in best_file_name:
            best_file.remove(best_file_name)
    if len(best_file) != 1:
        raise Exception ("too many or too few files found: {} found in {}".format(len(best_file), folder))

    with open(best_file[0]) as best_result_file:
        configuration['results']['best_network'] = copy.deepcopy(json.load(best_result_file))
    

    
                                
    configuration_top['configurations'].append(configuration)


print(json.dumps(configuration_top))

    
                                 
