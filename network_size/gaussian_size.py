



import sys
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
import os
import sobol
import resource
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""




def generate_sobol_points(M, dim):
    points = []
    for i in range(M):
        points.append(sobol.i4_sobol(dim,i)[0])
    return np.array(points)

dim = 6
M = int(2**16)

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

if len(sys.argv) != 3:
    print("Usage:\n\tpython {} number_of_widths number_of_heights\n\n".format(sys.argv[0]))
    exit(1)

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

def train(*, parameters, samples, title):
    train_sizes = [ 1024, 2048, 4096, 8192]
    
    optimizers = {"SGD": keras.optimizers.SGD}
    
    losses = ["mean_squared_error"]
    best_predictions = []
    for optimizer in optimizers.keys():
        for loss in losses:
            for train_size in train_sizes:
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

                prediction_errors = comm.gather(prediction_error, root=0)

                if rank == 0:
                    widths_all = base_width*2**np.arange(0, number_of_widths)
                    depths_all = base_depth*2**np.arange(0, number_of_depths)

                    w,d = np.meshgrid(widths_all, depths_all)
                    
                    prediction_errors = np.reshape(np.array(prediction_errors), w.shape)

                    print(prediction_errors)
                    plt.pcolormesh(w, d, prediction_errors)
                    
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.colorbar()
                    showAndSave.prefix='network_%s' % train_size
                    np.save('prediction_errors.npy', prediction_errors)
                    plt.savefig('prediction_error.png')
                    showAndSave('prediction_errors')
                    best_predictions=np.amin(prediction_errors)
    if rank == 0:
        plt.plot(train_sizes, best_predictions, '-*')
        plt.savefig('best_prediction.png')
        showAndSave('best_prediction')
    
            
    


# # Training

# In[4]:


for depth in depths:
    for width in widths:
        depth = int(depth)
        width=int(width)
        gaussian_network = [width for k in range(depth)]
        gaussian_network.append(1)
        title='{}_{}' .format (depth, width)
        train(parameters=parameters, samples=samples, title=title)
   

