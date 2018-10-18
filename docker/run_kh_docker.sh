#!/bin/bash
set -e
if [ -z "${AIRFOILS_DLMC_KH_DATAPATH}" ]
then
    echo 'You need to set ${AIRFOILS_DLMC_KH_DATAPATH} to a folder containing kh+_1.nc and qmc_points.txt'
    echo ''
    echo 'On Linux and OS X (and other versions of Bash on Windows), do:'
    echo ''
    echo '    export AIRFOILS_DLMC_KH_DATAPATH=/path/to/folder'
    echo ''
    echo 'then rerun this script'
    exit 1
fi

if [ ! -f ${AIRFOILS_DLMC_KH_DATAPATH}/kh_1.nc ];
then
    echo "AIRFOILS_DLMC_KH_DATAPATH (${AIRFOILS_DLMC_KH_DATAPATH}) does not contain:"
    echo "    kh_1.nc"
    echo "Please add the file to the folder and rerun this script"
    exit 2
fi

if [ ! -f ${AIRFOILS_DLMC_KH_DATAPATH}/qmc_points.txt ];
then
    echo "AIRFOILS_DLMC_KH_DATAPATH (${AIRFOILS_DLMC_KH_DATAPATH}) does not contain:"
    echo "    qmc_points.txt"
    echo "Please add the file to the folder and rerun this script"
    exit 3
fi

userid=$(id -u)
docker build . -t kjetilly/airfoils_learning:git
repodir=$(dirname $(pwd))

docker run --user ${userid} -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'pip3 freeze > /project/notebooks/pip_kh_packages.txt'

docker run --user ${userid} -v $repodir:/project -v ${AIRFOILS_DLMC_KH_DATAPATH}:/datakh --rm kjetilly/airfoils_learning:git bash -c 'export AIRFOILS_DLMC_KH_DATAPATH=/datakh; cd /project/notebooks && jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute KelvinHelmholtzMultipleSensors.ipynb --output KelvinHelmholtzMultipleSensorsOutput.ipynb'

