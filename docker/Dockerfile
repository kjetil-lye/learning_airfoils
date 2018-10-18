FROM ubuntu:18.04
LABEL learning_airfoils_version="git"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip

RUN pip3 install tensorflow keras jupyter[notebook] matplotlib matplotlib2tikz GitPython numpy scipy scikit-learn netcdf4 h5py
