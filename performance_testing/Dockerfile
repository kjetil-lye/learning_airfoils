ARG TENSORFLOW_VERSION="UNSET_PLEASE_SET_USING_BUILD_ARG"
ARG KERAS_VERSION="UNSET_PLEASE_SET_USING_BUILD_ARG"
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

RUN apt-get update && apt-get install --yes git python3-matplotlib python3-numpy libnetcdf-dev libhdf5-dev 
RUN pip install gitpython matplotlib2tikz h5py netcdf4 tabulate sobol sobol_seq
RUN cd /root &&\
    git clone https://github.com/keras-team/keras &&\
    cd keras &&\
    git checkout ${KERAS_VERSION} &&\
    python setup.py install
    


