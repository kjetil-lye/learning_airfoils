import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def get_tensorflow_versions():
    """
    See the dockerhub page for tensorlfow to get allowed tags
    https://hub.docker.com/r/tensorflow/tensorflow/tags
    """
    return ["1.12.0-py3",
            "1.11.0-py3",
            "1.10.0-py3",
            "1.9.0-py3",
            "1.7.1-py3",
            "1.7.0-py3"]

def run_command(cmd):
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise Exception("Failed running:\n{}".format(cmd))
    

def build_docker_with_tensorflow_version(version):
    run_command("docker build . -t kjetilly/airfoils:{tensorflow_version} --build-arg TENSORFLOW_VERSION='{tensorflow_version}'".format(tensorflow_version=version))

    

def run_with_tensorflow_version(version):
    run_command("docker run -v $(dirname $(pwd)):/learning --rm --user $(id --user) kjetilly/airfoils:{tensorflow_version} bash -c 'cd /learning/notebooks; python ../python/SodShockTubeQMC.py'".format(tensorflow_version=version))
            
if __name__ == '__main__':
    versions = get_tensorflow_versions()
    runtimes = []
    for version in versions:
        build_docker_with_tensorflow_version(version)
        start = time.time()
        run_with_tensorflow_version(version)
        end = time.time()

        runtime = end - start

        runtimes.append(runtime)

        with open("version_{}_runtime.txt".format(version)) as f:
            f.write(str(runtime))

    for version, runtime in zip(versions, runtimes):
        print("{}: {}".format(version, runtime))
    
              
              
              
    indices = np.arange(0, len(versions))

    plt.bar(indices, runtimes)
    plt.gca().set_xticks(indices)
    plt.gca().set_xticklabels(versions)
    plt.show()

    
