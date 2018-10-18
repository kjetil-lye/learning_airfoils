#!/bin/bash
set -e

docker build . -t kjetilly/airfoils_learning:git
repodir=$(dirname $(pwd))

docker run -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'pip3 freeze > /project/notebooks/pip_packages.txt'
docker run -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'cd /project/notebooks && jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute MachineLearningSixParametersAirfoil.ipynb --output MachineLearningSixParametersAirfoilOutput.ipynb'

