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
    regularizer,
    learning_rate,
    epochs):

    config_map = {
    "learning_rate" : learning_rate,
    "epochs" : epochs,
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





def submit(command, exports):
    exports = copy.deepcopy(exports)
    exports['MACHINE_LEARNING_DO_NOT_SAVE_PLOTS'] = 'on'
    exports['MACHINE_LEARNING_DO_NOT_PRINT_TABLES'] = 'on'
    exports['MACHINE_LEARNING_DO_NOT_SAVE_NP_DATA'] = 'on'
    export_str = " ".join("{}={}".format(k, exports[k]) for k in exports.keys())
    command_to_run = "{} bsub -n 1 -W 120:00 {}".format(export_str, command)
    with open('exports.sh', 'w') as exp_file:
        for k in exports.keys():
            exp_file.write("export {}={}\n".format(k, exports[k]))

    os.system(command_to_run)

def submit_notebook_in_parallel(notebook_name, depth, width, functional_name=None):
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
                            learning_rates = network_parameters.get_learning_rates()
                            for n, learning_rate in enumerate(learning_rates):
                                exports[network_parameters.get_learning_rates.key] = str(n)
                                epochs = network_parameters.get_epochs()
                                for m, epoch in enumerate(epochs):
                                    exports[network_parameters.get_epochs.key] = str(m)

                                    folder_name = "_".join([exports[k] for k in exports.keys()])
                                    folder_name = os.path.splitext(notebook_name)[0] +"_"+ folder_name
                                    folder_name = ''.join(ch for ch in folder_name if ch.isalnum() or ch =='_')
                                    print(folder_name)
                                    os.mkdir(folder_name)
                                    os.chdir(folder_name)
                                    print(os.getcwd())
                                    shutil.copyfile('../python/{}'.format(notebook_name), '{}'.format(notebook_name))
                                    os.mkdir('img')
                                    os.mkdir('img_tikz')
                                    os.mkdir('tables')
                                    os.mkdir('results')

                                    if functional_name is not None:
                                        notebook = "{} --functional_name {}".format(notebook, functional_name)
                                    submit('python {notebook}'.format(
                                        notebook = notebook
                                    ), exports)

                                    writeConfig(depth=depth,
                                        width=width,
                                        optimizer = optimizer,
                                        loss = loss,
                                        selection_type = selection_type,
                                        selection = selection,
                                        train_size = train_size,
                                        regularizer = regularizer,
                                        learning_rate = learning_rate,
                                        epochs=epoch)

                                    os.chdir('..')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Submit notebook in parallel')

    parser.add_argument('--script', default=None, help='The script to run', required=True)
    parser.add_argument('--number_of_widths', default=5, type=int, help='The number of widths to use')
    parser.add_argument('--number_of_depths', default=5, type=int, help='The number of depths to use')

    parser.add_argument('--functional_name',
                        default=None,
                        help='The functional to use options: depends on the script (optional)')

    args = parser.parse_args()
    functional_name = args.functional_name

    import sys
    notebook = parser.script

    notebook = os.path.basename(notebook)
    width = parser.number_of_widths
    depth = parser.number_of_depths

    print("Using depth = {}, width = {}".format(depth, width))
    submit_notebook_in_parallel(notebook, depth, width, functional_name = args.functional_name)
