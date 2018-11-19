import keras
import os


def get_optimizers():

    optimizers = {"Adam": keras.optimizers.Adam,
                "SGD": keras.optimizers.SGD}
    if get_optimizers.key in os.environ:
        optimizers = {os.environ[get_optimizers.key]: optimizers[os.environ[get_optimizers.key]]}
    return optimizers

get_optimizers.key = 'MACHINE_LEARNING_OPTIMIZER'
def get_losses():

    losses = ["mean_squared_error", 'mean_absolute_error']
    if get_losses.key in os.environ:
        losses = [losses[int(os.environ[get_losses.key])]]
    return losses
get_losses.key = 'MACHINE_LEARNING_LOSS'

def get_selections():

    selections = {}
    selections['Best performing'] = ['mean_train', 'ray_prediction', 'wasserstein_train', 'train']
    selections['Emperically optimal'] = ['mean', 'mean_tail', 'prediction', 'wasserstein']

    if get_selections.key in os.environ:
        selection_key = os.environ[get_selections.key]
        selection_map = {}
        for selection_type in selections.keys():
            for selection in selections[selection_type]:
                if selection == selection_key:
                    selection_map={selection_type:[selection_key]}
                    break
        selections = selection_map
    return selections

get_selections.key = 'MACHINE_LEARNING_SELECTION'

def get_training_sizes():
    sizes =  [128, 256, 512]
    if get_training_sizes.key in os.environ:
        sizes = [sizes[int(os.environ[get_training_sizes.key])]]
    return sizes

get_training_sizes.key = 'MACHINE_LEARNING_TRAINING_SIZE'

def get_regularizations(train_size):
    regularizations = [None,
        keras.regularizers.l2(0.01/train_size),
        keras.regularizers.l1(0.01/train_size)]
    if get_regularizations.key in os.environ:
        regularizers=[regularizers[int(os.environ[get_regularizations.key])]]
    return regularizations
get_regularizations.key = 'MACHINE_LEARNING_REGULARIZATION'
