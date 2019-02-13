#!/bin/bash
set -e

docker build . -t kjetilly/airfoils_learning:git
repodir=$(dirname $(pwd))
userid=$(id -u)
docker run --user ${userid} -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'cd /project/notebooks; bash run_speedtests.sh'
