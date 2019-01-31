#!/bin/bash
export OMP_NUM_THREADS=1
set -e
module load python/3.6.1

for func in 'Lift' 'Drag';
do

    bsub -W 48:00 -n 1 python3 ../python/ComputingBestNetworks.py --data_source 'Airfoils' --functional_name ${func};

done


for func in 'Q1' 'Q2' 'Q3';
do

    bsub -W 48:00 -n 1 python3 ../python/ComputingBestNetworks.py --data_source 'SodShockTubeQMC' --functional_name ${func};
done

for func in 'Sine' 'Sine/d' 'Sine/d3';
do

    bsub -W 48:00 -n 1 python3 ../python/ComputingBestNetworks.py --data_source 'Sine' --functional_name ${func};
done
