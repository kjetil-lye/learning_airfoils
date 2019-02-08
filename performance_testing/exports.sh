# We only want to train one specific network for this
# This corresponds to
# {
#  "learning_rate": 0.01,
#  "epochs": 500000,
#  "number_of_depths": 1,
#  "number_of_widths": 1,
#  "optimizer": "Adam",
#  "loss": "mean_squared_error",
#  "train_size": 128,
#  "selection_type": "Best performing",
#  "selction": "wasserstein_train",
#  "regularizer": {
#    "l1": 0,
#    "l2": 7.81249980263965e-07
#  }
#}

export MACHINE_LEARNING_NUMBER_OF_WIDTHS=1
export MACHINE_LEARNING_NUMBER_OF_DEPTHS=1
export MACHINE_LEARNING_OPTIMIZER=Adam
export MACHINE_LEARNING_LOSS=0
export MACHINE_LEARNING_SELECTION=wasserstein_train
export MACHINE_LEARNING_TRAINING_SIZE=2
export MACHINE_LEARNING_REGULARIZATION=3
export MACHINE_LEARNING_LEARNING_RATE=0
export MACHINE_LEARNING_EPOCHS=0
export MACHINE_LEARNING_DO_NOT_SAVE_PLOTS=on
export MACHINE_LEARNING_DO_NOT_PRINT_TABLES=on
export MACHINE_LEARNING_DO_NOT_SAVE_NP_DATA=on
