#!/bin/bash
# Runs the effective network for each functional

# This corresponds to 128 training samples
export MACHINE_LEARNING_TRAINING_SIZE=2
export OMP_NUM_THREADS=1
set -e


python3 ../python/split_best_networks_into_individual_files.py
python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'Airfoils' --functional_name Drag --functional_name Lift
python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q1 --functional_name Q2 --functional_name Q3
