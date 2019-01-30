# coding: utf-8


import numpy.random
import tensorflow
import os
import random
def seed_random_number(seed):
    # see https://stackoverflow.com/a/52897216
    numpy.random.seed(seed)
    tensorflow.set_random_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
random_seed = 42


seed_random_number(random_seed)
#####################
from keras import backend as K
# see https://stackoverflow.com/a/52897216 we really need singlethread to get
# reproducible results!
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
K.set_session(sess)
#####################
import sys
import time

from plot_info import *
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import matplotlib
import matplotlib.pyplot as plt
import netCDF4
import os.path
import scipy
import scipy.stats

import os
from print_table import *
import resource
import gc
def print_memory_usage():
    gc.collect()


    print("Memory usage: %s" %  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class NetworkInformation(object):
    def __init__(self, *, optimizer, epochs, network,
                 train_size, validation_size,
                 activation='relu',
                 error_length=1000,
                 loss='mean_squared_error',
                 tries=5,
                 selection='train',
                 large_integration_points=None,
                 activity_regularizer=None,
                 kernel_regularizer = None,
                 learning_rate=0.01,
                 monte_carlo_parameters = None,
                 monte_carlo_values = None):
        self.network = network
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_size = train_size
        self.validation_size = validation_size
        self.activation=activation
        self.error_length = 1000
        self.loss = loss
        self.tries=tries
        self.selection=selection
        self.large_integration_points = large_integration_points
        self.activity_regularizer = activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.learning_rate = learning_rate
        self.monte_carlo_parameters = monte_carlo_parameters
        self.monte_carlo_values = monte_carlo_values



class Tables(object):
    def __init__(self, tables):
        self.tables = tables

    def write_tables(self):
        for k in self.tables.keys():
            self.tables[k].print_table(k)

    def __getitem__(self, key):
        return self.tables[key]

    @staticmethod
    def make_default():
        tables = {}
        tables['speedup_table'] = TableBuilder()
        tables['comparison_table'] = TableBuilder()
        tables['wasserstein_table_builder'] = TableBuilder()
        tables['bilevel_speedup_table']= TableBuilder()
        tables['prediction_error_table'] = TableBuilder()

        return Tables(tables)


class OutputInformation(object):
    def __init__(self, *, tables, title, short_title,
                enable_plotting=True, enable_tables= True,
                sampling_method='QMC'):
        self.tables = tables
        self.title = title
        self.short_title = short_title
        self.enable_plotting = enable_plotting
        self.enable_tables = enable_tables
        self.sampling_method = sampling_method


    def write_tables(self):
        self.tables.write_tables()


def compute_prediction_error(data, data_predicted, train_size, norm_ord):
    base = np.linalg.norm(data, ord=norm_ord)/data.shape[0]

    if base < 0.001:
        base = 1

    diff = np.linalg.norm(data[train_size:]-data_predicted[train_size:], ord=norm_ord)/(data.shape[0]-train_size)

    return diff / base

def get_network(parameters, data, *, network_information, output_information):
    train_size = network_information.train_size

    validation_size = network_information.validation_size
    title = output_information.title
    short_title = output_information.short_title

    optimizer = network_information.optimizer
    epochs = network_information.epochs


    input_size = parameters.shape[1]

    best_network = None
    best_network_index = None
    best_learning_rate = None


    tries = 5
    start_total_learning = time.time()

    y_train = data[:train_size]
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    lsq_predictor = reg.fit(parameters[:train_size,:], y_train)

    for trylearn in range(network_information.tries):
        model = Sequential()
        model.add(Dense(network_information.network[0],
            input_shape=(input_size,),
            activation=network_information.activation,
            kernel_regularizer=network_information.kernel_regularizer))
        for layer in network_information.network[1:-1]:
            model.add(Dense(layer, activation=network_information.activation, kernel_regularizer = network_information.kernel_regularizer))
        model.add(Dense(network_information.network[-1],
            activity_regularizer=network_information.activity_regularizer,
            kernel_regularizer=network_information.kernel_regularizer))




        model.compile(optimizer=optimizer(lr=network_information.learning_rate),
                      loss=network_information.loss)

        weights = np.copy(model.get_weights())

        x_train = parameters[:train_size,:]

        y_train = data[:train_size]


        x_val = parameters[train_size:validation_size+train_size,:]
        y_val = data[train_size:train_size+validation_size]

        epochs_r=range(1, epochs)

        start_training_time = time.time()

        if 'ray' not in network_information.selection:
            hist = model.fit(x_train, y_train, batch_size=train_size, epochs=epochs,shuffle=True,
                         validation_data=(x_val, y_val),verbose=0)
        else:
            training_ray_samples = int(0.7*train_size)
            validation_ray_samples = train_size - training_ray_samples
            hist = model.fit(x_train[:training_ray_samples,:], y_train[:training_ray_samples], batch_size=train_size, epochs=epochs,shuffle=True,
                         validation_data=(x_train[training_ray_samples:, :], y_train[training_ray_samples:]),verbose=0)
        print()
        end_training_time = time.time()


        print("Training took {} seconds".format(end_training_time-start_training_time))
        console_log("Training took {} seconds".format(end_training_time-start_training_time))

        if network_information.selection == 'train':
            train_error = np.sum(hist.history['loss'][-min(network_information.error_length,epochs):])
        elif network_information.selection == 'prediction':
            start_prediction_time = time.time()
            train_error = compute_prediction_error(data, np.reshape(model.predict(parameters), data.shape), train_size, 2)#np.sum(np.linalg.norm(data - np.reshape(model.predict(parameters), data.shape), ord=2))/data.shape[0]

            end_prediction_time = time.time()
            print("Prediction error computation took: {} seconds".format(end_prediction_time-start_prediction_time))

        elif network_information.selection == 'mean_tail':
            train_error = abs(np.sum(data)/data.shape[0] - np.sum(np.reshape(model.predict(parameters), data.shape))/data.shape[0])
        elif network_information.selection == 'mean':
            qmc_means = []
            dlqmc_means = []

            for k in range(1, data.shape[0]):
                qmc_means.append(np.sum(data[:k])/k)
                dlqmc_means.append(np.sum(np.reshape(model.predict(parameters[:k,:]), k))/k)

            qmc_means = np.array(qmc_means)
            dlqmc_means = np.array(dlqmc_means)

            train_error = np.sum(abs(qmc_means-dlqmc_means))
        elif network_information.selection =='mean_train':
            qmc_means = []
            dlqmc_means = []

            for k in range(1, train_size):
                qmc_means.append(np.sum(data[:k])/k)
                dlqmc_means.append(np.sum(np.reshape(model.predict(parameters[:k,:]), k))/k)

            qmc_means = np.array(qmc_means)
            dlqmc_means = np.array(dlqmc_means)
            train_error = np.sum(abs(qmc_means-dlqmc_means))

        elif network_information.selection == 'wasserstein':
            train_error = scipy.stats.wasserstein_distance(data, reshape(model.predict(parameters), data.shape))

        elif network_information.selection == 'wasserstein_train':
            train_error = scipy.stats.wasserstein_distance(data[:train_size], reshape(model.predict(parameters[:train_size,:]), train_size))

        elif network_information.selection == 'ray_prediction':
            train_error = compute_prediction_error(y_train, \
                np.reshape(model.predict(parameters[:train_size,:]), train_size), training_ray_samples, 2)#np.sum(np.linalg.norm(data - np.reshape(model.predict(parameters), data.shape), ord=2))/data.shape[0]

        else:
            raise Exception("Unknown selection %s" % network_information.selection)

        # we want to measure the variance at some point
        compute_stats_with_reuse(model, lsq_predictor, network_information, output_information, parameters, data, train_size, "try_{}".format(trylearn))
        if best_network is None or train_error < best_learning_rate:
            best_network = model
            best_network_index = trylearn
            best_learning_rate = train_error
            best_weights = weights
        if output_information.enable_plotting:
            plt.loglog(hist.history['loss'],label="Training loss")
            plt.loglog(hist.history['val_loss'], label='Validation loss')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("Training and validation loss\n%s\n(epochs=%d)" % (title, epochs))
            showAndSave('dummy')
        np.save("results/" + showAndSave.prefix + "training_losses_%d.npy" % trylearn, hist.history['loss'])
        np.save("results/" + showAndSave.prefix + "validation_losses_%d.npy" % trylearn, hist.history['val_loss'])
        gc.collect()



    output_information.selection_error = best_learning_rate
    end_total_learning = time.time()


    print("Best network index: %d" % best_network_index)
    console_log("Best network index: %d" % best_network_index)
    print("Total learning time took: %d s" % (end_total_learning-start_total_learning))
    console_log("Total learning time took: %d s" % (end_total_learning-start_total_learning))
    model = best_network
    weights = best_weights

    print_keras_model_as_table('network', model)
    # save model to file, see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_json = model.to_json()
    with open("results/" + showAndSave.prefix + "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("results/" + showAndSave.prefix + "model.h5")
    np.save("results/" + showAndSave.prefix + "intial.npy", weights)

    #plt.loglog(hist.history['loss'])
    #plt.title("Training loss\n%s\n(epochs=%d)" % (title, epochs))
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #showAndSave("training_loss")

    #plt.loglog(hist.history['val_loss'])
    #plt.title("Validation loss\n%s\n(epochs=%d)" % (title, epochs))
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #showAndSave("validation_loss")


    end_training_time = time.time()
    print("Training took {} seconds".format(end_training_time-start_training_time))
    if output_information.enable_plotting:

        plt.loglog(hist.history['loss'],label="Training loss")
        plt.loglog(hist.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training and validation loss\n%s\n(epochs=%d)" % (title, epochs))
        showAndSave("training_validation_loss")

    x_test = parameters
    y_test = data
    y_predict = model.predict(x_test)

    if output_information.enable_plotting:
        plt.scatter(y_test[train_size:], y_predict[train_size:,0])
        plt.title("Scatter plot, \n%s,\n epochs=%d" % (title, epochs))
        plt.xlabel("Actual data")
        plt.ylabel("Predicted data")
        showAndSave("scatter_ml")

    print("Number of parameters: %d"% model.count_params())

    gc.collect()

    return  model, data, parameters




def get_network_and_postprocess(parameters, samples, *, network_information,
    output_information):

    sampling_method = output_information.sampling_method


    network, data, parameters = get_network(parameters, samples,
        network_information=network_information,
        output_information=output_information)

    title = output_information.title
    train_size = network_information.train_size
    epochs = network_information.epochs
    model = network
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    y_train = data[:train_size]
    coeffs = reg.fit(parameters[:train_size,:], y_train)

    evaluated_lsq = coeffs.predict(parameters)

    if output_information.enable_plotting:
        plt.scatter(data[train_size:], evaluated_lsq[train_size:])
        plt.title('Linear least squares\n%s' % title)
        plt.xlabel("Actual data")
        plt.ylabel("Interpolated data")
        showAndSave('scatter_lsq')

    def myvar(x):
        mean = sum(x)/x.shape[0]
        var = sum((mean-x)**2)/x.shape[0]
        return var

    def mymean (x):
        return sum(x)/x.shape[0]



    wasserstein_table_builder = output_information.tables['wasserstein_table_builder']
    bilevel_speedup_table = output_information.tables['bilevel_speedup_table']
    prediction_error_table = output_information.tables['prediction_error_table']
    comparison_table = output_information.tables['comparison_table']

    speedup_table = output_information.tables['speedup_table']

    variance_top = myvar(data)

    predicted = network.predict(parameters)
    predicted = predicted.reshape(parameters.shape[0])
    variance_diff_ml = myvar(data- predicted)


    bilevel_speedup_table.set_header(["Functional", "DLb{} Speedup".format(output_information.sampling_method)])
    bilevel_speedup_table.add_row([output_information.short_title, variance_top/variance_diff_ml])

    variance_diff_interpolate =myvar(data - evaluated_lsq)

    mean_qmc = mymean(data)

    mean_ml = mymean(network.predict(parameters))

    mean_few_qmc = mymean(parameters[:train_size,:])






    predicted_all = network.predict(parameters)
    predicted_all = predicted_all.reshape(parameters.shape[0])
    print(predicted_all.shape)
    mean_mlmlmc = mymean(predicted[:train_size]-data[:train_size]) + mymean(predicted_all)


    var_qmc = myvar(data)

    var_ml = myvar(network.predict(parameters))

    var_few_qmc = myvar(parameters[:train_size,:])


    print(parameters.shape)
    gc.collect()
    try:

        if output_information.enable_plotting:
            plt.hist(data,bins=40,density=True,label='{} {} samples'.format(sampling_method,
                samples.shape[0]),alpha=0.5)
            plt.title("Comparison %s and DL%s\n%s\nepochs=%d"% (sampling_method, sampling_method,title, epochs))
            plt.hist(network.predict(parameters),bins=40,density=True,
            label='DL%s (%d samples)' % (sampling_method, train_size),alpha=0.5)
            plt.legend()
            showAndSave('hist_qmc_ml')


            plt.title("Comparison %s with %d and %s with %d samples\n%s" %(sampling_method,8192, sampling_method, train_size, title))
            plt.hist(data,bins=40,density=True,label='{} {} samples'.format(sampling_method, samples.shape[0]),alpha=0.5)
            plt.hist(data[:train_size],bins=40,density=True, alpha=0.5,
            label='%s %d samples' % (sampling_method, train_size))
            plt.legend()
            showAndSave('hist_qmc_qmc')

            plt.title("Comparison %s with least squares\n%s" % (sampling_method, title))
            plt.hist(data,bins=40,density=True,label='{} {} samples'.format(sampling_method, samples.shape[0]),alpha=0.5)
            plt.hist(evaluated_lsq,bins=40,density=True,alpha=0.5,
                label='Least squares (%d points)' % train_size)
            plt.legend()
            showAndSave('hist_qmc_lsq')
    except Exception as e:
        print(e)

    if network_information.large_integration_points is not None:
        print("Computing large integration points")
        if output_information.enable_plotting:
            plt.hist(data,bins=40,density=True,label='{} {} samples'.format(sampling_method, samples.shape[0]),
                alpha=0.5)
            plt.title("Comparison %s and DL%s (large integration points with %d points)\n%s\nepochs=%d"% (sampling_method, sampling_method, network_information.large_integration_points.shape[0], title, epochs))
            plt.hist(network.predict(network_information.large_integration_points),bins=40,density=True,
                     label='DL%s (%d samples)' % (sampling_method, train_size),alpha=0.5)
            plt.legend()
            showAndSave('hist_qmc_ml_large')



    prediction_error_table.set_header(["Functional", "Deep learning", "Least squares"])
    norms = [1, 2]
    norm_names = ["$L^1$", "$L^2$"]

    print_memory_usage()
    output_information.prediction_error = {}
    for norm, norm_name in zip(norms, norm_names):
        prediction_error = compute_prediction_error(data, np.reshape(model.predict(parameters), data.shape), train_size, norm)

        output_information.prediction_error[norm] = prediction_error
        prediction_error_lsq = compute_prediction_error(data, np.reshape(evaluated_lsq, data.shape),train_size, norm)



        prediction_error_table.add_row(["{} ({})".format(output_information.short_title,
            norm_name), prediction_error, prediction_error_lsq])

    print_memory_usage()



    gc.collect()

    samples = 2**np.arange(2,int(log2(data.shape[0]))+1)
    if samples[-1] != data.shape[0]:
        samples = [k for k in samples]
        samples.append(data.shape[0])
        samples=np.array(samples)

    stats = {}
    for stat in ['mean', 'var']:
        gc.collect()
        stats[stat]={}
        stats[stat]['sources']={}
        if stat == 'mean':
            stats[stat]['compute']=lambda x: sum(x)/x.shape[0]
        else:
            stats[stat]['compute']=lambda x: sum(x**2)/x.shape[0]-(sum(x)/x.shape[0])**2


        stats[stat]['sources'][sampling_method]={}
        stats[stat]['sources']['DL%s' % sampling_method] = {}
        stats[stat]['sources']['Least squares'] = {}
        stats[stat]['sources']['DLb%s' % sampling_method] = {}

        stats[stat]['sources'][sampling_method]['data']=array([stats[stat]['compute'](data[:k]) for k in samples])
        stats[stat]['sources']['DL%s' % sampling_method]['data'] = array([stats[stat]['compute'](array(model.predict(parameters[:k,:]))) for k in samples])
        stats[stat]['sources']['Least squares']['data'] = array([stats[stat]['compute'](evaluated_lsq[:k]) for k in samples])

        stats[stat]['sources']['DLb%s' % sampling_method]['data'] = [0]

        for k in samples[1:]:
            if stat == 'mean':
                mean = mymean(data[:train_size]-predicted[:train_size]) + mymean(predicted_all)


                stats[stat]['sources']['DLb%s' % sampling_method]['data'].append(mean)
            elif stat=='var':
                mean = mymean(data[:train_size]-predicted[:train_size]) + mymean(predicted_all)

                m2 = mymean(data[:train_size]**2-predicted[:train_size]**2) + mymean(predicted_all**2)

                stats[stat]['sources']['DLb%s' % sampling_method]['data'].append(m2-mean**2)

        stats[stat]['sources']['DLb%s' % sampling_method]['data']=array(stats[stat]['sources']['DLb%s' % sampling_method]['data'])

        sources = stats[stat]['sources'].keys()
        for source in sources:

            stats[stat]['sources'][source]['representative'] = stats[stat]['sources'][source]['data'][-1]




        if output_information.enable_plotting:
            for source in stats[stat]['sources'].keys():
                if 'DLb%s' % sampling_method not in source:
                    plt.plot(samples, stats[stat]['sources'][source]['data'], label=source)
            plt.xlabel('Number of samples ($J_L$)')
            plt.ylabel('%s' % stat)
            plt.title('%s as a function of number of samples used for evaluation\n%s' % (stat, title))
            plt.legend()
            showAndSave('function_of_samples_%s'  % (stat))
        stats[stat]['sources']['%s %d' % (sampling_method, train_size)] = {}
        stats[stat]['sources']['%s %d' % (sampling_method, train_size)]['representative'] = stats[stat]['compute'](data[:train_size])

    sources = [source for source in stats['mean']['sources'].keys()]
    datatable = [[],[],[]]
    print_memory_usage()
    for source in sources:
        datatable[0].append(source)
    comparison_table.set_upper_header(datatable[0])

    for source in sources:
        for stat in ['mean', 'var']:
            datatable[1].append(stat)
    comparison_table.set_lower_header(datatable[1])

    for source in sources:
        for stat in ['mean', 'var']:
            datatable[2].append(stats[stat]['sources'][source]['representative'])


    comparison_table.add_row([output_information.short_title]+datatable[2])

    #### Speedup
    speeduptable = [[],[],[]]
    statstouse = ['mean', 'var']
    baseline=sampling_method
    small_baseline = '%s %d' % (sampling_method, train_size)
    competitors = ['%s %d' % (sampling_method, train_size), 'DL%s' % sampling_method, 'DLb%s'% sampling_method, 'Least squares']

    for competitor in competitors:
        speeduptable[0].append(competitor)

    speedup_table.set_upper_header(speeduptable[0])

    for source in competitors:
        for stat in ['mean', 'var']:
            speeduptable[1].append(stat)

    speedup_table.set_lower_header(speeduptable[1])


    for competitor in competitors:
        for stat in ['mean', 'var']:



            error = abs(stats[stat]['sources'][competitor]['representative']-stats[stat]['sources'][baseline]['representative'])

            error_base = abs(stats[stat]['sources'][small_baseline]['representative']-stats[stat]['sources'][baseline]['representative'])

            speedup = error_base/error

            speeduptable[2].append(speedup)
    speedup_table.add_row([output_information.short_title] + speeduptable[2])
    output_information.stat_error = {}

    for stat in ['mean', 'var']:
        output_information.stat_error[stat] = float(abs(stats[stat]['sources']['DL%s' % sampling_method]['representative']-stats[stat]['sources'][baseline]['representative'])/abs(stats[stat]['sources'][baseline]['representative']))
        output_information.stat_error['%s_comparison' % stat] =float( abs(stats[stat]['sources'][small_baseline]['representative']-stats[stat]['sources'][baseline]['representative'])/abs(stats[stat]['sources'][baseline]['representative']))
    for stat in ['mean', 'var']:
        errors_qmc = []


        for k in range(len(samples[1:-2])):
            errors_qmc.append(abs(stats[stat]['sources'][sampling_method]['representative']- stats[stat]['sources'][sampling_method]['data'][k+1]))

        if output_information.enable_plotting:
            plt.loglog(samples[1:-2], errors_qmc, '-o', label='%s error' % sampling_method)
            plt.axvline(x=train_size, linestyle='--', color='grey')

            for competitor in ['DLb%s' % sampling_method, 'DL%s' % sampling_method, 'Least squares']:
                error = abs(stats[stat]['sources'][competitor]['representative']-stats[stat]['sources'][baseline]['representative'])



                plt.loglog(samples[1:-2], error*ones_like(samples[1:-2]), '--', label='%s error' % competitor)

            plt.xlabel('Number of samples for %s' % sampling_method)
            plt.ylabel('Error')
            plt.title('Error for %s compared to %s\n%s' % (stat, sampling_method, title))
            plt.legend()
            showAndSave("error_evolution_%s" % stat)



    ###### Speedup Wasserstein

    data_modified = data
    if int(log2(data_modified.shape[0])) != log2(data_modified.shape[0]):
        if float(4**int(log2(data_modified.shape[0])) - data_modified.shape[0]) < 0.1 * data_modified.shape[0]:
            # if the data size is not a power of two, we pad the array with the
            # same value at the end.
            data_modified_tmp = []

            for d in data_modified:
                data_modified_tmp.append(d)
            for k in range(2*2**int(log2(data_modified.shape[0]))-data_modified.shape[0]):
                data_modified_tmp.append(data_modified[-1])
            data_modified = np.array(data_modified_tmp)
        else:
            data_modified = data_modified[:2**int(data_modified.shape[0])]
    N_wasser = 2**(int(log2(data_modified.shape[0])))
    data_wasser = data_modified[:N_wasser]
    qmc_upscaled = repeat(data[:train_size], N_wasser/train_size)

    wasser_qmc_qmc = scipy.stats.wasserstein_distance(data_wasser, qmc_upscaled)
    wasser_qmc_ml = scipy.stats.wasserstein_distance(data_wasser, reshape(model.predict(parameters), data.shape))
    wasser_qmc_lsq = scipy.stats.wasserstein_distance(data_wasser, evaluated_lsq)

    speedup_qmc = wasser_qmc_qmc / wasser_qmc_qmc
    speedup_ml =  wasser_qmc_qmc / wasser_qmc_ml
    speedup_lsq = wasser_qmc_qmc / wasser_qmc_lsq

    wasserstein_table=[["DL%s" % sampling_method, "Least squares", "%s %d" % (sampling_method, train_size)],[speedup_ml, speedup_lsq, speedup_qmc]]


    wasserstein_table_builder.set_header(wasserstein_table[0])
    wasserstein_table_builder.add_row([output_information.short_title]+wasserstein_table[1])



    output_information.stat_error['wasserstein'] = wasser_qmc_ml
    output_information.stat_error['wasserstein_comparison'] = wasser_qmc_qmc
    errors_qmc = []

    for k in range(1, int(log2(data_modified.shape[0]))):
        qmc_upscaled = repeat(data_modified[:int(2**k)], N_wasser//int(2**k))
        errors_qmc.append(scipy.stats.wasserstein_distance(data_wasser, qmc_upscaled))

    samples_wasser = 2**array(range(1, int(log2(data_modified.shape[0]))))

    if output_information.enable_plotting:
        plt.loglog(samples_wasser, errors_qmc, '-o', label='%s error' % sampling_method)

        plt.loglog(samples_wasser, wasser_qmc_ml*ones_like(samples_wasser), '--', label='DL%s error' % sampling_method)
        plt.loglog(samples_wasser, wasser_qmc_lsq*ones_like(samples_wasser), '--', label='LSQ error')
        plt.axvline(x=train_size, linestyle='--', color='grey')
        plt.xlabel('Number of samples for %s' % sampling_method)
        plt.ylabel('Error (Wasserstein)')
        plt.title('Error (Wasserstein) compared to %s\n%s' % (sampling_method, title))
        plt.legend()
        showAndSave("error_evolution_wasserstein")



    if network_information.large_integration_points is not None:
        qmc_large = network_information.large_integration_points
        errors_qmc = []

        for k in range(1, int(log2(data_modified.shape[0]))):
            qmc_upscaled = repeat(data_modified[:int(2**k)], N_wasser//int(2**k))
            errors_qmc.append(scipy.stats.wasserstein_distance(data_wasser, qmc_upscaled))

        samples_wasser = 2**array(range(1, int(log2(data_modified.shape[0]))))

        evaluated_lsq_large = coeffs.predict(qmc_large)

        print("Trying with a large number of %s samples %d" % (sampling_method, qmc_large.shape[0]))
        large_predicted = np.reshape(model.predict(qmc_large), qmc_large.shape[0])
        data_wasser_large = np.repeat(data_wasser, qmc_large.shape[0]/data_wasser.shape[0])

        wasser_qmc_ml = scipy.stats.wasserstein_distance(data_wasser_large, large_predicted)
        wasser_qmc_lsq = scipy.stats.wasserstein_distance(data_wasser_large, evaluated_lsq_large)

        speedup_qmc = wasser_qmc_qmc / wasser_qmc_qmc
        speedup_ml =  wasser_qmc_qmc / wasser_qmc_ml
        speedup_lsq = wasser_qmc_qmc / wasser_qmc_lsq

        if output_information.enable_plotting:
            plt.loglog(samples_wasser, errors_qmc, '-o', label='%s error' % sampling_method)

            plt.loglog(samples_wasser, wasser_qmc_ml*ones_like(samples_wasser), '--', label='DL%s error' % sampling_method)
            plt.loglog(samples_wasser, wasser_qmc_lsq*ones_like(samples_wasser), '--', label='LSQ error')
            plt.axvline(x=train_size, linestyle='--', color='grey')
            plt.xlabel('Number of samples for %s' % sampling_method)
            plt.ylabel('Error (Wasserstein)')
            plt.title('Error (Wasserstein) compared to %s (using more samples)\n%s' % (sampling_method, title))
            plt.legend()
            showAndSave("error_evolution_wasserstein_large")


    compute_stats_with_reuse(network, coeffs, network_information, output_information, parameters, data, train_size)

    console_log("done one configuration")

    return network, data, parameters


def plot_train_size_convergence(network_information,
                                output_information,
                                run_function,
                                max_size):
    sampling_method = output_information.sampling_method
    train_sizes = 2**np.arange(2, int(log2(max_size)))
    errors = {}
    errors_comparison={}
    error_keys = ['prediction L2', 'wasserstein', 'mean', 'var']
    for error_key in error_keys:
        errors[error_key]=[]
        errors_comparison[error_key]=[]
    for train_size in train_sizes:
        tables = Tables.make_default()
        batch_size=train_size
        validation_size=train_size
        network_information.train_size = train_size
        network_information.validation_size = train_size
        network_information.batch_size = train_size
        output_information.enable_plotting = False

        print_comparison_table.silent = True
        seed_random_number(random_seed)
        run_function(network_information, output_information)

        errors['prediction L2'].append(output_information.prediction_error[2])
        for stat in ['wasserstein', 'mean', 'var']:
            errors[stat].append(output_information.stat_error[stat])
            errors_comparison[stat].append(output_information.stat_error['%s_comparison'%stat])


    showAndSave.silent=False
    for error_key in error_keys:
        error = errors[error_key]

        plt.loglog(train_sizes, error, '-o',label='DL%s %s' % (sampling_method, error_key))

        print("errors[%s] = [%s]" % (error_key, ", ".join(["%.16f" % k for k in error])))

        if 'prediction' not in error_key:
            comparison_error = errors_comparison[error_key]
            plt.loglog(train_sizes, comparison_error, '-o',label='%s %s' % (sampling_method, error_key))
        plt.legend()

        plt.xlabel('Training size')
        plt.ylabel("Error")
        plt.title("Error of %s as a function of training size" % error_key)
        showAndSave('convergence_%s' % error_key)

def replace_data_with_actual_data(evaluated_data, real_data, train_size):
    """ Used in compute_stats_with_reuse"""
    new_data = np.reshape(copy.deepcopy(evaluated_data), evaluated_data.shape[0])
    new_data[:train_size] = real_data[:train_size]
    return new_data

def add_actual_data(evaluated_data, real_data, train_size):
    """ Used in compute_stats_with_reuse"""
    new_data = np.zeros(evaluated_data.shape[0]+train_size)
    new_data[:train_size] = real_data[:train_size]
    new_data[train_size:] = evaluated_data

    return new_data

def wasserstein_error_cut(x, all_data):
    if x.shape[0] > all_data.shape[0]:
        return wasserstein_error_cut(all_data, x)

    repeat_number = int(all_data.shape[0]/x.shape[0])

    N = x.shape[0] * repeat_number

    cut_data = all_data[:N]

    return scipy.stats.wasserstein_distance(cut_data, np.repeat(x, repeat_number))


def wasserstein_error_extend(x, all_data):
    if x.shape[0] > all_data.shape[0]:
        return wasserstein_error_extend(all_data, x)

    repeat_number = int(ceil(all_data.shape[0]/x.shape[0]))



    N = x.shape[0] * repeat_number


    extended_data = np.zeros(N)
    extended_data[:all_data.shape[0]] = all_data
    for k in range (N-all_data.shape[0]):
        extended_data[all_data.shape[0]+k] = all_data[-1]


    return scipy.stats.wasserstein_distance(extended_data, np.repeat(x, repeat_number))

import json
import sobol

def mean_bilevel(x, all_data, train_size):
    return np.mean(all_data[:min(train_size, x.shape[0])]-x[:min(train_size, x.shape[0])])+np.mean(x)

def var_bilevel(x, all_data, train_size):
    return np.mean(all_data[:min(train_size, x.shape[0])]**2-x[:min(train_size, x.shape[0])]**2)+np.mean(x**2)-mean_bilevel(x, all_data, train_size)**2

def var_bilevel_alternative(x, all_data, train_size):
    m = mean_bilevel(x, all_data, train_size)
    return np.mean((all_data[:min(train_size, x.shape[0])]-m)**2-(x[:min(train_size, x.shape[0])]-m)**2)+np.mean((x-m)**2)


def compute_stats_with_reuse(network, lsq_predictor, network_information, output_information, parameters, data, train_size, postfix=""):
    dim = parameters.shape[1]
    M_self_generated = 2**15
    parameter_sources = {
        '{}_from_data'.format(output_information.sampling_method): parameters,
        'QMC' : np.array([sobol.i4_sobol(dim, p)[0] for p in range(M_self_generated)]),
        'MC' : np.random.uniform(0,1,(M_self_generated, dim))
        }

    if network_information.large_integration_points is not None:
        parameter_sources['{}_from_data_large'.format(output_information.sampling_method)] = network_information.large_integration_points




    modifiers = {
        'ordinary' : lambda evaluated_data, real_data, train_size: evaluated_data,
        'replace' : replace_data_with_actual_data,
        'add' : add_actual_data,
        'remove' : lambda evaluated_data, real_data, train_size : evaluated_data[train_size:]
    }

    errors_functionals = {
        'mean_error' : lambda x, all_data: abs(np.mean(x)-np.mean(all_data)),
        'var_error' : lambda x, all_data: abs(np.var(x)-np.var(all_data)),
        'mean_error_relative' : lambda x, all_data: abs(np.mean(x)-np.mean(all_data))/abs(np.mean(all_data)),
        'var_error_relative' : lambda x, all_data: abs(np.var(x)-np.var(all_data))/abs(np.var(all_data)),
        'speedup_mean_bilevel' : lambda x, all_data: abs(np.var(all_data)/np.var(all_data[:min(train_size, x.shape[0])]-x[:min(train_size, x.shape[0])])),
        'mean_bilevel_error' : lambda x, all_data: abs(mean_bilevel(x, all_data, train_size) - np.mean(all_data)),
        'var_bilevel_error' :  lambda x, all_data: abs(var_bilevel(x, all_data, train_size) - np.var(all_data)),
        'var_bilevel_error_alternative' :  lambda x, all_data: abs(var_bilevel_alternative(x, all_data, train_size) - np.var(all_data)),
        'mean' : lambda x, all_data: float(np.mean(x)),
        'var' : lambda x, all_data: float(np.var(x)),
        'mean_bilevel' : lambda x, all_data : float(mean_bilevel(x, all_data, train_size)),
        'var_bilevel' : lambda x, all_data : float(var_bilevel(x, all_data, train_size)),
        'var_bilevel_alternative' : lambda x, all_data : float(var_bilevel_alternative(x, all_data, train_size)),
        'wasserstein_error_cut' : wasserstein_error_cut,
        'wasserstein_error_extend' : wasserstein_error_extend,
        'prediction_l1_relative' : lambda x, all_data: float(compute_prediction_error(all_data[:min(x.shape[0], all_data.shape[0])], x[:min(x.shape[0], all_data.shape[0])], 0, 1)),
        'prediction_l2_relative' : lambda x, all_data: float(compute_prediction_error(all_data[:min(x.shape[0], all_data.shape[0])], x[:min(x.shape[0], all_data.shape[0])], 0, 2))
    }

    predictors = {
        'ml' : lambda x: np.reshape(network.predict(x), x.shape[0]),
        'lsq' : lambda x: np.reshape(lsq_predictor.predict(x), x.shape[0])
    }

    results = {}
    for parameter_source in parameter_sources.keys():
        current_data = parameter_sources[parameter_source]
        results[parameter_source] = {}
        for predictor in predictors.keys():
            predicted_data = predictors[predictor](current_data)
            results[parameter_source][predictor] = {}
            for modifier in modifiers.keys():
                modified_data = modifiers[modifier](predicted_data, data, train_size)
                results[parameter_source][predictor][modifier] = {}
                for error_functional in errors_functionals.keys():

                    results[parameter_source][predictor][modifier][error_functional] = errors_functionals[error_functional](modified_data, data)
    all_results_with_information = {}
    all_results_with_information['algorithms'] = results
    all_results_with_information['train_size'] = train_size
    all_results_with_information['base_sampling_error']  = {}
    for error_functional in errors_functionals:
        all_results_with_information['base_sampling_error'][error_functional] = errors_functionals[error_functional](data[:train_size], data)

    all_results_with_information['reference_sampling_error']  = {}
    for error_functional in errors_functionals:
        all_results_with_information['reference_sampling_error'][error_functional] = errors_functionals[error_functional](data, data)


    norm_names = ["L1","L2"]
    norms=[1,2]
    all_results_with_information['prediction_error']  = {}
    all_results_with_information['prediction_error']['ML']  = {}
    all_results_with_information['prediction_error']['LSQ']  = {}
    for norm, norm_name in zip(norms, norm_names):
        all_results_with_information['prediction_error']['ML'][norm_name] = compute_prediction_error(data, np.reshape(network.predict(parameters), data.shape), train_size, norm)


        all_results_with_information['prediction_error']['LSQ'][norm_name] = compute_prediction_error(data, np.reshape(lsq_predictor.predict(parameters), data.shape),train_size, norm)




    all_results_with_information['mc_speedup'] = {}
    all_results_with_information['mc_errors'] = {}


    if network_information.monte_carlo_parameters is not None:
        mc_parameters = network_information.monte_carlo_parameters
        mc_values = network_information.monte_carlo_values

        parameter_sources = {
            '{}_from_data'.format(output_information.sampling_method): parameters,
            'MC' : mc_parameters
            }

        for parameter_source_name in parameter_sources.keys():
            all_results_with_information['mc_speedups'][parameter_source_name] = {}
            all_results_with_information['mc_errors'][parameter_source_name] = {}
            predicted_data = predictors['ml'](parameter_sources[parameter_source_name])
            all_results_with_information['mc_errors'][parameter_source_name]['mc_base_error'] = {}
            for modifier in modifiers.keys():
                all_results_with_information['mc_speedup'][parameter_source_name][modifier] = {}
                all_results_with_information['mc_errors'][parameter_source_name][modifier] = {}

                modified_data = modifiers[modifier](predicted_data, mc_values, train_size)
                for error_functional in errors_functionals.keys():
                    error = errors_functionals[error_functional](modified_data, mc_values)
                    base_error = errors_functionals[error_functional](mc_values[:train_size], mc_values)
                    all_results_with_information['mc_errors'][parameter_source_name][modifier][error_functional] = error
                    try:
                        all_results_with_information['mc_speedup'][parameter_source_name][modifier][error_functional] = base_error / error
                    except:
                        all_results_with_information['mc_speedup'][parameter_source_name][modifier][error_functional] = 0.1
            all_results_with_information['mc_errors'][parameter_source_name]['mc_base_error'][error_functional] = base_error
            table_speedup = TableBuilder()
            table_speedup.set_header(["Error name", *modifiers.keys()])

            for error_functional in errors_functionals.keys():
                row = [error_functional]

                for modifier in modifiers.keys():
                    row.append("{:.3f}".format(all_results_with_information['mc_speedup'][parameter_source_name][modifier][error_functional]))
                table_speedup.add_row(row)

            table_speedup.set_title("Speedup of Machine learning compared to Monte Carlo for various errors and tactics, with {epochs}, {learning_rate}, using {parameter_source_name} as parameters".format(epochs=network_information.epochs,
                learning_rate=network_information.learning_rate,
                parameter_source_name=parameter_source_name))
            table_speedup.print_table("mc_speedups_{}".format(parameter_source_name.lower()))


            table_error = TableBuilder()
            table_error.set_header(["Error name", *modifiers.keys()])

            for error_functional in errors_functionals.keys():
                row = [error_functional]

                for modifier in modifiers.keys():
                    row.append("{:.3f}".format(all_results_with_information['mc_error'][parameter_source_name][modifier][error_functional]))
                table_error.add_row(row)

            table_error.set_title("Errors of Machine learning compared to Monte Carlo for various errors and tactics, with {epochs}, {learning_rate}, using {parameter_source_name} as parameters".format(epochs=network_information.epochs,
                learning_rate=network_information.learning_rate,
                parameter_source_name=parameter_source_name))
            table_error.print_table("mc_errors_{}".format(parameter_source_name.lower()))


    if len(postfix) > 0 and postfix[0] != "_":
        postfix= "_" + postfix
    with open('results/' + showAndSave.prefix + '_combination_stats{}.json'.format(postfix), 'w') as f:
        json.dump(all_results_with_information, f)

    console_log(json.dumps(all_results_with_information))
