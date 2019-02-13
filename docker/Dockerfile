ARG TENSORFLOW_VERSION="1.12.0"

FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

RUN apt-get update && apt-get install --yes git python3-matplotlib python3-numpy libnetcdf-dev libhdf5-dev 
RUN pip install gitpython matplotlib2tikz h5py netcdf4 tabulate sobol sobol_seq keras

    


