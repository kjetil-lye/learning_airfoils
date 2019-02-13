#!/bin/bash
# This corresponds to 128 training samples
export MACHINE_LEARNING_TRAINING_SIZE=2
export MACHINE_LEARNING_DO_NOT_SAVE_PLOTS=on
export OMP_NUM_THREADS=1
set -e
python3 ../python/split_best_networks_into_individual_files.py

for network in 'Lift' 'Drag' 'effective'
do
    python3 ../python/speedtest.py --json_file ../data/${network}_best_network.json --data_source 'Airfoils' --functional_name Lift --functional_name Drag
done

for network in 'Q1' 'Q2' 'Q3' 'effective'
do
  python3 ../python/speedtest.py --json_file ../data/${network}_best_network.json --data_source 'SodShockTubeQMC' --functional_name Q1 --functional_name Q2 --functional_name Q3
done
