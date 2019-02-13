ARG TENSORFLOW_VERSION="1.12.0-py3"

FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

RUN apt-get update && apt-get install --yes git python3-matplotlib python3-numpy libnetcdf-dev libhdf5-dev python3-pip 
RUN pip3 install gitpython matplotlib2tikz h5py netcdf4 tabulate sobol sobol_seq keras tabulate jupyter

ARG USERID='1000'
RUN useradd -r -u ${USERID}  appuser
RUN mkdir /home/appuser
RUN chown appuser /home/appuser
USER appuser
    


