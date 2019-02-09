import network_parameters
import os
import shutil
import json
import copy
import glob
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





def submit(command, exports, arguments):
    exports = copy.deepcopy(exports)
    exports['MACHINE_LEARNING_DO_NOT_SAVE_PLOTS'] = 'on'
    exports['MACHINE_LEARNING_DO_NOT_PRINT_TABLES'] = 'on'
    exports['MACHINE_LEARNING_DO_NOT_SAVE_NP_DATA'] = 'on'
    export_str = " ".join("{}={}".format(k, exports[k]) for k in exports.keys())
    command_to_run = "{} bsub -n 1 -W 120:00 {}".format(export_str, command)
    with open('exports.sh', 'w') as exp_file:
        for k in exports.keys():
            exp_file.write("export {}={}\n".format(k, exports[k]))

    if arguments is not None:
        command_to_run = "{} {}".format(command_to_run, arguments)

    os.system(command_to_run)

def submit_notebook_in_parallel(notebook_name, depth, width, functional_name=None, only_missing=False, prefix=''):
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
                                    folder_name = os.path.splitext(notebook_name)[0] +prefix + "_"+ folder_name
                                    folder_name = ''.join(ch for ch in folder_name if ch.isalnum() or ch =='_')
                                    print(folder_name)
                                    if not only_missing:
                                        os.mkdir(folder_name)
                                    os.chdir(folder_name)
                                    print(os.getcwd())

                                    should_run = not only_missing
                                    if only_missing:
                                        lsf_files = glob.glob('lsf.*')
                                        if len(lsf_files)==0:
                                            should_run = True

                                        else:
                                            with open(lsf_files[0]) as lsf_file:
                                                lsf_content = lsf_file.read()

                                                if 'Successfully completed' not in lsf_content:
                                                    should_run = True

                                    if should_run:
                                        shutil.copyfile('../python/{}'.format(notebook_name), '{}'.format(notebook_name))
                                        if not only_missing:
                                            os.mkdir('img')
                                            os.mkdir('img_tikz')
                                            os.mkdir('tables')
                                            os.mkdir('results')

                                        arguments = None
                                        if functional_name is not None:
                                            arguments = "--functional_name {}".format(functional_name)
                                        submit('python {notebook}'.format(
                                            notebook = notebook_name
                                        ), exports, arguments)

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

    parser.add_argument('--only_missing',  action='store_true', help='Only submit configurations that did not complete')

    parser.add_argument('--functional_name',
                        default=None,
                        help='The functional to use options: depends on the script (optional)')

    args = parser.parse_args()
    functional_name = args.functional_name

    import sys
    notebook = args.script

    notebook = os.path.basename(notebook)
    width = args.number_of_widths
    depth = args.number_of_depths
    if functional_name is not None:
        prefix = "_{}".format(functional_name)
    else:
        prefix = ""

    print("Using depth = {}, width = {}".format(depth, width))
    submit_notebook_in_parallel(notebook, depth, width,
        functional_name = args.functional_name,
        only_missing=args.only_missing,
        prefix=prefix)
