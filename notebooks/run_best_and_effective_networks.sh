#!/bin/bash
# This corresponds to 128 training samples
export MACHINE_LEARNING_TRAINING_SIZE=2
export OMP_NUM_THREADS=1
set -e
python3 ../python/split_best_networks_into_individual_files.py

for network in 'Lift' 'Drag' 'effective'
do
  bash -c "python3 ../python/ComputingBestNetworks.py --json_file ../data/${network}_best_network.json --data_source 'Airfoils' --functional_name Lift --functional_name Drag &> log_best_${network}_airfoils.txt";
done

for network in 'Q1' 'Q2' 'Q3' 'effective'
do
  bash -c "python3 ../python/ComputingBestNetworks.py --json_file ../data/${network}_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q1 --functional_name Q2 --functional_name Q3 &> log_best_${network}_sod.txt";
done

for network in 'effective';
do
    bash -c "python3 ../python/ComputingBestNetworks.py --json_file ../data/${network}_best_network.json --data_source 'Sine' --functional_name 'Sine' --functional_name 'Sine/d' --functional_name 'Sine/d3' &> log_best_${network}_sine.txt";
done
