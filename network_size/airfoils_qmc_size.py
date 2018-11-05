import os
import sobol
import resource
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




import sys

sys.path.append('../python')
sys.path.append('.')
from network_size import *


from machine_learning import *

import numpy as np


class TrainingFunction(object):
    def __init__(self, *, parameters, samples):
        self.parameters = parameters
        self.samples=samples


    def __call__(self, network_information, output_information):
        showAndSave.prefix='%s_%s_%s_ts_%d_bs_%d' %(title,
            network_information.optimizer.__name__,
            network_information.loss,
            network_information.batch_size,
            network_information.train_size)

        get_network_and_postprocess(self.parameters, self.samples,
                    network_information = network_information,
                    output_information = output_information)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:\n\tpython {} number_of_widths number_of_heights\n\n".format(sys.argv[0]))
        exit(1)


    qmc_points = np.loadtxt('../sobol_6_8000.txt')
    qmc_points = qmc_points[1:].reshape((8000,6))

    large_qmc_points = np.loadtxt('../sobol_6_131072.txt')
    all_points = qmc_points.copy()
    forces = np.array(np.loadtxt('../force_6_params.dat'))


    N = min(qmc_points.shape[0], forces.shape[0])
    qmc_points = qmc_points[:N,:]
    forces  = forces[:N,:]


    input_size=6
    train_size=128
    validation_size=128

    training_sizes = 2**np.arange(3, 10)

    number_of_widths=int(sys.argv[1])
    number_of_depths = int(sys.argv[2])

    epochs = 500000


    force_names=['Lift', 'Drag']
    for n, force_name in enumerate(force_names):
        parameters = qmc_points
        samples = forces[:, n+1]
        run_function = TrainingFunction(parameters=parameters,
            samples = samples)


        optimizers = {"SGD": keras.optimizers.SGD}

        losses = ["mean_squared_error"]

        optimizer = 'SGD'
        loss = losses[0]
        tables = Tables.make_default()

        network_information = NetworkInformation(optimizer=optimizers[optimizer], epochs=epochs,
                                                 network=None, train_size=None,
                                                 validation_size=None,
                                                loss=loss, tries=5, selection='prediction')


        title = force_name
        short_title = force_name
        output_information = OutputInformation(tables=tables, title=title,
                                              short_title=title, enable_plotting=False)


        find_best_network_size(network_information = network_information,
            output_information = output_information,
            training_sizes = training_sizes,
            run_function = run_function,
            number_of_depths = number_of_depths,
            number_of_widths = number_of_widths,
            base_title = title)




# # Training

# In[4]:
