import os
import sobol
import resource
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""





import sys
if len(sys.argv) != 4:
    print("Usage:\n\tpython {} number_of_widths number_of_heights dimension\n\n".format(sys.argv[0]))
    exit(1)


sys.path.append('../python')

import matplotlib
from mpi4py import MPI
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from machine_learning import *
import tensorflow as tf
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import scipy.stats



def generate_sobol_points(M, dim):
    points = []
    for i in range(M):
        points.append(sobol.i4_sobol(dim,i)[0])
    return np.array(points)

dim = int(sys.argv[3])
M = int(2**20)

data_sources = {"QMC Sobol" : generate_sobol_points(M, dim)}


def sine_functional(x):
    return np.sum(np.sin(4*np.pi*x), 1)
def normal_functional(x):
    return scipy.stats.norm.ppf(x)

def sum_functional(x):
    return np.sum(x, 1)

functionals = {
               "Sine" : sine_functional
               }


number_of_widths=int(sys.argv[1])
number_of_depths = int(sys.argv[2])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if int(np.sqrt(size)) != np.sqrt(size):
    print("number of MPI processors needs to be a square number")
    exit(1)
number_of_procs_for_width = int(np.sqrt(size))
number_of_procs_for_depth = int(np.sqrt(size))
number_of_widths_per_proc = number_of_widths / number_of_procs_for_width
number_of_depths_per_proc = number_of_depths / number_of_procs_for_depth

start_width = number_of_widths_per_proc*(rank % number_of_procs_for_width)
start_depth = number_of_depths_per_proc*(rank // number_of_procs_for_width)

base_width=6
base_depth=4
widths = base_width*2**np.arange(start_width, start_width + number_of_widths_per_proc)
depths = base_depth*2**np.arange(start_depth, start_depth + number_of_depths_per_proc)






epochs = 500


parameters = data_sources['QMC Sobol']
samples = functionals['Sine'](parameters)

def train(*, parameters, samples, title, train_size):


    optimizers = {"SGD": keras.optimizers.SGD}

    losses = ["mean_squared_error"]
    best_predictions = []
    optimizer = 'SGD'
    loss = losses[0]
    tables = Tables.make_default()
    batch_size = train_size
    validation_size=train_size



    network_information = NetworkInformation(optimizer=optimizers[optimizer], epochs=epochs,
                                             network=gaussian_network, train_size=train_size,
                                             validation_size=validation_size,
                                            loss=loss, tries=5, selection='prediction')

    output_information = OutputInformation(tables=tables, title=title,
                                          short_title=title)
    showAndSave.prefix='%s_%s_%s_ts_%d_bs_%d' %(title, optimizer, loss, batch_size, train_size)
    get_network_and_postprocess(parameters, samples, network_information = network_information,
        output_information = output_information)

    showAndSave.prefix='%s_%s_%s_all_ts_%d_bs_%d' %(title, optimizer, loss, batch_size, train_size)
    tables.write_tables()

    prediction_error = output_information.prediction_error[2]

    return prediction_error






# # Training

# In[4]:
all_depths = base_depth * 2**np.arange(0, number_of_depths)
all_widths = base_width * 2**np.arange(0, number_of_widths)
best_predictions = []
tain_sizes = 2**np.arange(3,12)
for train_size in train_sizes:
    prediction_errors = np.zeros((len(depths), len(widths)))
    for depth in depths:
        for width in widths:
            depth = int(depth)
            width=int(width)
            gaussian_network = [width for k in range(depth)]
            gaussian_network.append(1)
            title='{}_{}' .format (depth, width)
            prediction_errors[depth, width] = train(parameters=parameters, samples=samples, title=title)

    # doing this the safe way
    comm.barrier()
    prediction_errors_all = None
    if rank == 0:
        prediction_errors_all = np.zeros((number_of_depths, number_of_widths))

    for n in range(len(depths)):
        for m in range(len(widths)):
            # This could probably have been done with one gather, but doing it
            # the stupid way to make sure it is correct
            sub_errors = comm.gather([depths[n], widths[m], prediction_errors[n,m]],
                root=0)

            if rank == 0:
                for error_pair in sub_errors:
                    depth = error_pair[0]
                    width = error_pair[1]
                    error = error_pair[2]

                    i = all_depths.index(depth)
                    j = all_widths.index(width)

                    prediction_errors_all[i,j] = error

    if rank == 0:
        w,d = np.meshgrid(all_widths, all_widths)

        plt.pcolormesh(w, d, prediction_errors_all)

        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar()
        showAndSave.prefix='network_%s' % train_size
        np.save(showAndSave.prefix + '_prediction_errors.npy', prediction_errors)

        showAndSave('prediction_errors')
        best_predictions.append(np.amin(prediction_errors_all))
        print_memory_usage()

if rank == 0:
    plt.loglog(train_sizes, best_predictions, '-o')
    plt.xlabel('Training size')
    plt.ylabel("Best prediction error")

    showAndSave.prefix = 'network_'
    showAndSave('best_network')
    print(best_predictions)
