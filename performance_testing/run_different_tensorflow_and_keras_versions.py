import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append('../python')
from plot_info import showAndSave, savePlot

def get_tensorflow_versions():
    """
    See the dockerhub page for tensorlfow to get allowed tags
    https://hub.docker.com/r/tensorflow/tensorflow/tags
    """
    return ["1.12.0-py3",
            "1.11.0-py3",
            "1.10.0-py3",
            "1.9.0-py3",
            "1.7.1-py3"]


def get_keras_versions():
    """
    see the github release page
    
    https://github.com/keras-team/keras/releases
    
    (we are skipping releases which has a bug fix release a couple of days afterwards)
    """
    return ["2.2.4",
            "2.2.2",
            "2.2.0",
            "2.1.6",
            "2.1.5",
            "2.1.4",
            "2.1.3"]
            
def run_command(cmd):
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise Exception("Failed running:\n{}".format(cmd))
    

def build_docker_with_keras_and_tensorflow_version(*, tensorflow_version,
                                                   keras_version):
    run_command("docker build . -t kjetilly/airfoils:{tensorflow_version}_{keras_version} --build-arg TENSORFLOW_VERSION='{tensorflow_version}' --build-arg KERAS_VERSION='{keras_version}'".format(tensorflow_version=tensorflow_version, keras_version=keras_version))

    

def run_with_keras_and_tensorflow_version(*, tensorflow_version, keras_version):
    
    run_command("docker run -v $(dirname $(pwd)):/learning --rm --user $(id --user) kjetilly/airfoils:{tensorflow_version}_{keras_version} bash -c 'cd /learning/notebooks; source ../performance_testing/exports.sh; python ../python/SodShockTubeQMC.py'".format(tensorflow_version=tensorflow_version, keras_version=keras_version))
            
if __name__ == '__main__':
    versions = get_tensorflow_versions()
    default_keras_version = get_keras_versions()[0]
    runtimes = []
    for version in versions:
        build_docker_with_keras_and_tensorflow_version(tensorflow_version = version, keras_version= default_keras_version)
        start = time.time()
        run_with_keras_and_tensorflow_version(tensorflow_version = version, keras_version = default_keras_version)
        end = time.time()

        runtime = end - start

        runtimes.append(runtime)

        with open("tensorflow_version_{}_keras_version_{}_runtime.txt".format(version, default_keras_version), "w") as f:
            f.write(str(runtime))

    for version, runtime in zip(versions, runtimes):
        print("{}: {}".format(version, runtime))
    
              
              
              
    indices = np.arange(0, len(versions))

    plt.bar(indices, runtimes)
    plt.gca().set_xticks(indices)
    plt.gca().set_xticklabels(versions)
    plt.xlabel("Tensorflow version")
    plt.ylabel("Total runtime")
    plt.title("Runtimes for different tensorflow version\nEach runtime includes 5 retrainings with an (approx) 12*10 network,\nand 5 retrainings with a 6x4 network\nand done for three functionals (Q1, Q2 and Q3)\nKeras version: {}".format(default_keras_version))
    savePlot('runtimes_tensorflow')
    plt.close()

    keras_versions = get_keras_versions()
    default_tensorflow_version = get_tensorflow_versions()[0]
    runtimes_keras = []
    for keras_version in keras_versions:
        build_docker_with_keras_and_tensorflow_version(tensorflow_version = default_tensorflow_version,
                                             keras_version=keras_version)
        start = time.time()
        run_with_keras_and_tensorflow_version(tensorflow_version = default_tensorflow_version,
                                    keras_version = keras_version)
        end = time.time()

        runtime = end - start

        runtimes_keras.append(runtime)

        with open("tensorflow_version_{}_keras_version_{}_runtime.txt".format(default_tensorflow_version, keras_version), "w") as f:
            f.write(str(runtime))

    for version, runtime in zip(keras_versions, runtimes_keras):
        print("{}: {}".format(version, runtime))
    
              
              
              
    indices = np.arange(0, len(keras_versions))

    plt.bar(indices, runtimes)
    plt.gca().set_xticks(indices)
    plt.gca().set_xticklabels(keras_versions)
    plt.xlabel("Keras version")
    plt.ylabel("Total runtime")
    plt.title("Runtimes for different Keras versions\nEach runtime includes 5 retrainings with an (approx) 12*10 network,\nand 5 retrainings with a 6x4 network\nand done for three functionals (Q1, Q2 and Q3)\nTensorflow version: {}".format(default_tensorflow_version))
    showAndSave('runtimes_keras')
    
