import network_parameters
import os
import shutil
import json
import copy
def writeConfig(*,
    depth,
    width,
    optimizer,
    loss,
    selection_type,
    selection,
    train_size,
    regularizer):

    config_map = {
    "number_of_depths" : depth,
    "number_of_widths" : width,
    "optimizer" : optimizer,
    "loss" : loss,
    "train_size" : train_size,
    "selection_type" : selection_type,
    "selction" : selection,
    "train_size" : train_size,
    "regularizer" : regularizer.get_config() if regularizer is not None else "None"
    }

    with open("config_run.json", "w") as outfile:
        json.dump(config_map, outfile)



def downloaded_needed_packages():
    if not os.path.exists("../packages_downloaded"):
        os.mkdir("../packages_downloaded")
    if not downloaded_needed_packages.has_run:
        os.system("HOME=$(dirname $(pwd))/packages_downloaded pip install --user matplotlib matplotlib2tikz gitpython tabulate sobol sobol_seq")
        downloaded_needed_packages.has_run = True
    os.system("cp -r ../packages_downloaded/* ./")

downloaded_needed_packages.has_run = False



def submit(command, exports):
    exports = copy.deepcopy(exports)
    exports['HOME'] = os.getcwd()
    exports['JUPYTER_PATH'] = os.getcwd()
    exports['JUPYTER_CONFIG_DIR'] = os.getcwd()
    exports['JUPYTER_RUNTIME_DIR'] = os.getcwd()
    export_str = " ".join("{}={}".format(k, exports[k]) for k in exports.keys())
    command_to_run = "{} bsub -n 1 -W 120:00 {}".format(export_str, command)
    with open('exports.sh', 'w') as exp_file:
        for k in exports.keys():
            exp_file.write("export {}={}\n".format(k, exports[k]))
    # First we install the packages needed
    old_home = os.environ["HOME"]

    command_to_run.replace("$HOME", old_home)
    downloaded_needed_packages()
    os.system(command_to_run)

def submit_notebook_in_parallel(notebook_name, depth, width):
    exports = {}
    exports['MACHINE_LEARNING_NUMBER_OF_WIDTHS'] = str(width)
    exports["MACHINE_LEARNING_NUMBER_OF_DEPTHS"] = str(depth)

    for optimizer in network_parameters.get_optimizers().keys():
        exports[network_parameters.get_optimizers.key] = optimizer
        for i, loss in enumerate(network_parameters.get_losses()):
            exports[network_parameters.get_losses.key] = str(i)
            selections = network_parameters.get_selections()
            for selection_type in selections.keys():
                for selection in selections[selection_type]:
                    exports[network_parameters.get_selections.key] = selection

                    train_sizes = network_parameters.get_training_sizes()

                    for j, train_size in enumerate(train_sizes):
                        exports[network_parameters.get_training_sizes.key] = str(j)
                        regularizers = network_parameters.get_regularizations(train_size)

                        for k, regularizer in enumerate(regularizers):
                            exports[network_parameters.get_regularizations.key] = str(k)

                            folder_name = "_".join([exports[k] for k in exports.keys()])
                            folder_name = os.path.splitext(notebook_name)[0] +"_"+ folder_name
                            folder_name = ''.join(ch for ch in folder_name if ch.isalnum() or ch =='_')
                            print(folder_name)
                            os.mkdir(folder_name)
                            os.chdir(folder_name)
                            print(os.getcwd())
                            shutil.copyfile('../notebooks/{}'.format(notebook_name), '{}'.format(notebook_name))
                            os.mkdir('img')
                            os.mkdir('img_tikz')
                            os.mkdir('tables')
                            os.mkdir('results')
                            output = notebook.replace('.ipynb', 'Output.ipynb')
                            submit('$HOME/.local/bin/jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute {notebook} --output {output}'.format(
                                notebook = notebook,
                                output = output
                            ), exports)

                            writeConfig(depth=depth,
                                width=width,
                                optimizer = optimizer,
                                loss = loss,
                                selection_type = selection_type,
                                selection = selection,
                                train_size = train_size,
                                regularizer = regularizer)

                            os.chdir('..')

if __name__ == '__main__':
    import sys
    notebook = sys.argv[1]

    notebook = os.path.basename(notebook)
    width = 5
    depth = 5
    if len(sys.argv) == 4:

        width = int(sys.argv[2])
        depth = int(sys.argv[3])

    print("Using depth = {}, width = {}".format(depth, width))
    submit_notebook_in_parallel(notebook, depth, width)
