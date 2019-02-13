#!/bin/bash
set -e
userid=$(id -u)
docker build . -t kjetilly/airfoils_learning:git --build-arg USERID=${userid}

repodir=$(dirname $(pwd))

docker run --user ${userid} -v $repodir:/project --rm kjetilly/airfoils_learning:git bash -c 'cd /project/notebooks; bash run_speedtests.sh'
