# Using docker to run these experiments

The ```Dockerfile``` in this subfolder can be used to generate a docker image that can be used run all experiments.

## Getting Docker

[Consult the online documentation](https://docs.docker.com/install/). Install the CE (Community Edition) version.

These files were last tested with

    Docker version 18.09.1-ce, build 4c52b901c6

## Getting your user id

On Linux, you do

    userid=$(id -u)

for other operating system, consult the manual.

## Building the docker container

Use

    docker build . -t airfoils_learning:git --build-arg USERID=${userid}

where ```${userid}``` is the user id of your user.

## Running experiments

User

    docker run --user ${userid} --rm  -v PATH_TO_GIT_REPO:/project \
       airfoils_learning:git bash -c 'cd /project/notebooks; bash <script to run>'

where ```PATH_TO_GIT_REPO``` is the local path to this git repo (base folder) and ```<script to run>``` is the name of the (shell) script you wish to run. Replace

    bash <script to run>

with

    python <python script to run>

in the last part if it is a python script. ```${userid}``` is the user id.


## Running the notebook

Run

    docker run -p 8888:8888 --user ${userid} --rm -v PATH_TO_GIT_REPO:/project \
       airfoils_learning:git bash -c 'cd /project/notebooks; jupyter notebook'

where ```PATH_TO_GIT_REPO``` is the local path to this git repo (base folder). ```${userid}``` is the user id.