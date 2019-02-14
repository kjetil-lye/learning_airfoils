#!/bin/bash

# Runs the network size analysis through docker

set -e
userid=$(id -u)
docker build . -t kjetilly/airfoils_learning:git --build-arg USERID=${userid}

repodir=$(dirname $(pwd))

docker run --rm --user ${userid} -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'cd /project/notebooks; bash run_network_size_analysis.sh'
