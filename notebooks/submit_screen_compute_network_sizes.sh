#!/bin/bash
# Runs the effective network for each functional

# This corresponds to 128 training samples
export MACHINE_LEARNING_TRAINING_SIZE=2
export OMP_NUM_THREADS=1
set -e


python3 ../python/split_best_networks_into_individual_files.py
screen -S "airfoilslift" -dm bash -c "python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'Airfoils' --functional_name Lift"

screen -S "airfoilsdrag" -dm bash -c "python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'Airfoils' --functional_name Drag"

screen -S "sodshocktubeq1" -dm bash -c "python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q1"

screen -S "sodshocktubeq2" -dm bash -c "python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'SodShockTubeQMC'  --functional_name Q2"

screen -S "sodshocktubeq3" -dm bash -c "python3 ../python/ComputingBestNetworks.py --try_network_sizes --json_file ../data/effective_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q3"
