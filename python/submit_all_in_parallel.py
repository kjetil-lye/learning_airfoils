import network_parameters
import os
import shutil

def submit(command, exports):
    export_str = " ".join("{}={}".format(k, exports[k]) for k in exports.keys())
    command_to_run = "{} bsub -n 1 -W 120:00 {}".format(export_str, command)
    with open('exports.sh', 'w') as exp_file:
        for k in exports.keys():
            exp_file.write("export {}={}\n".format(k, exports[k]))
    os.system(command_to_run)

def submit_notebook_in_parallel(notebook_name):
    exports = {}
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
                            os.chdir('..')

if __name__ == '__main__':
    import sys
    notebook = sys.argv[1]

    notebook = os.path.basename(notebook)

    submit_notebook_in_parallel(notebook)
