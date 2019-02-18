#!/bin/bash
# Runs the best networks for each functional, together with the effective network

# This corresponds to 128 training samples
export MACHINE_LEARNING_TRAINING_SIZE=2
export OMP_NUM_THREADS=1
set -e


python3 ../python/split_best_networks_into_individual_files.py

for network in 'Lift' 'Drag'
do
    python3 ../python/ComputingBestNetworks.py --do_not_train_single_size_first --load_weights --json_file ../data/${network}_best_network.json --data_source 'Airfoils' --functional_name ${network} 
done

python3 ../python/ComputingBestNetworks.py --do_not_train_single_size_first --load_weights --json_file ../data/effective_best_network.json --data_source 'Airfoils' --functional_name Drag --functional_name Lift

for network in 'Q1' 'Q2' 'Q3' 
do
  python3 ../python/ComputingBestNetworks.py --do_not_train_single_size_first --load_weights --json_file ../data/${network}_best_network.json --data_source 'SodShockTubeQMC' --functional_name ${network}
done

python3 ../python/ComputingBestNetworks.py --do_not_train_single_size_first --load_weights --json_file ../data/effective_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q1 --functional_name Q2 --functional_name Q3
