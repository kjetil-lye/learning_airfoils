#!/bin/bash
# Submits alls script on euler. 



# EDIT THIS LINE TO THE CORRECT LOCATION
export AIRFOILS_DLMC_KH_DATAPATH=/cluster/project/sam/klye/kh_airfoils_data/

if [[ ! -d ${AIRFOILS_DLMC_KH_DATAPATH} ]];
then
    echo "The exported variable AIRFOILS_DLMC_KH_DATAPATH does not point to an existing folder"
    echo "Please update $0 and set it to a folder containing kh_1.nc and qmc_large.txt"
    exit 1
fi

if [ ! -f ${AIRFOILS_DLMC_KH_DATAPATH}/kh_1.nc ];
then
    echo "The exported variable AIRFOILS_DLMC_KH_DATAPATH does not contain kh_1.nc"
    echo "Please update $0 and set it to a folder containing kh_1.nc and qmc_large.txt"
    exit 1
fi

if [ ! -f ${AIRFOILS_DLMC_KH_DATAPATH}/qmc_large.txt ];
then
    echo "The exported variable AIRFOILS_DLMC_KH_DATAPATH does not contain qmc_large.txt"
    echo "Please update $0 and set it to a folder containing kh_1.nc and qmc_large.txt"
    exit 1
fi

# Accept additional parameters from the command line. This could eg be "--only_missing"
additional_parameters=$@

# Abort on first error found
set -e

for q in Lift Drag; 
do 
    python python/submit_all_pure_python_in_parallel.py --functional_name ${q} --script python/MachineLearningSixParametersAirfoil.py --number_of_widths 3 --number_of_depths 3 ${additional_parameters}; 
done; 

for q in Lift Drag; 
do 
    python python/submit_all_pure_python_in_parallel.py --functional_name ${q} --script python/MachineLearningSixParametersAirfoilMonteCarlo.py --number_of_widths 3 --number_of_depths 3 ${additional_parameters}; 
done; 

for q in Q1 Q2 Q3; 
do 

    python python/submit_all_pure_python_in_parallel.py --functional_name ${q} --script python/SodShockTubeQMC.py --number_of_widths 3 --number_of_depths 3 ${additional_parameters}; 
done ; 

for q in Sine Sine/d Sine/d3; 
do 
    python python/submit_all_pure_python_in_parallel.py --functional_name ${q} --script python/GaussianRandomVariable.py --number_of_widths 3 --number_of_depths 3 ${additional_parameters}; 
done;

for q in Q1 Q2; 
do 
    python python/submit_all_pure_python_in_parallel.py --functional_name ${q} --script python/KelvinHelmholtzMultipleSensors.py --number_of_widths 3 --number_of_depths 3 ${additional_parameters}; 
done
