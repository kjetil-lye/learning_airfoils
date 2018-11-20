import network_parameters
import os
import shutil
import json
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





def submit(command, exports):
    export_str = " ".join("{}={}".format(k, exports[k]) for k in exports.keys())
    command_to_run = "{} bsub -n 1 -W 120:00 {}".format(export_str, command)
    with open('exports.sh', 'w') as exp_file:
        for k in exports.keys():
            exp_file.write("export {}={}\n".format(k, exports[k]))
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
