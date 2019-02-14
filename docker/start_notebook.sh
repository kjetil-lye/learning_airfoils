#!/bin/bash
set -e
userid=$(id -u)

docker build . -t kjetilly/airfoils_learning:git --build-arg USERID=${userid}

repodir=$(dirname $(pwd))
docker run -p 8888:8888 --user ${userid} --rm -v $(dirname $(pwd)):/project \
       kjetilly/airfoils_learning:git bash -c 'cd /project/notebooks; jupyter notebook --ip 0.0.0.0 --port 8888'

