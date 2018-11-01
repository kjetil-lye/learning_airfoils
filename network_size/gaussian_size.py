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

import scipy.stats



def generate_sobol_points(M, dim):
    points = []
    for i in range(M):
        points.append(sobol.i4_sobol(dim,i)[0])
    return np.array(points)

def sine_functional(x):
    return np.sum(np.sin(4*np.pi*x), 1)

def square_functional(x):
    return np.sum(x**2, 1)

def circle_functional(x):
    return np.sum((x-0.5)**4, 1)
def normal_functional(x):
    return scipy.stats.norm.ppf(x)

def sum_functional(x):
    return np.sum(x, 1)


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
    if len(sys.argv) != 4:
        print("Usage:\n\tpython {} number_of_widths number_of_heights dimension\n\n".format(sys.argv[0]))
        exit(1)

    dim = int(sys.argv[3])
    M = int(2**18)

    training_sizes = 2**np.arange(3, int(np.log2(M)-5))

    data_sources = {"QMC Sobol" : generate_sobol_points(M, dim)}


    functionals = {"sine" : sine_functional,
                   "square" : square_functional,
                   "circle" : circle_functional}


    number_of_widths=int(sys.argv[1])
    number_of_depths = int(sys.argv[2])

    epochs = 500000


    for functional_name in functionals:
        parameters = data_sources['QMC Sobol']
        samples = functionals[functional_name](parameters)

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


        title = functional_name
        short_title = functional_name
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
