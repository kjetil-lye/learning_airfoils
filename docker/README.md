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
       airfoils_learning:git bash -c 'cd /project/notebooks; jupyter notebook --ip 0.0.0.0 --port 8888'

where ```PATH_TO_GIT_REPO``` is the local path to this git repo (base folder). ```${userid}``` is the user id.

You will get out of the form

    [I 10:18:20.751 NotebookApp] Writing notebook server cookie secret to /home/appuser/.local/share/jupyter/runtime/notebook_cookie_secret
    [I 10:18:20.902 NotebookApp] Serving notebooks from local directory: /project/notebooks
    [I 10:18:20.902 NotebookApp] The Jupyter Notebook is running at:
    [I 10:18:20.902 NotebookApp] http://(eefe0a64fbf9 or 127.0.0.1):8888/?token=ffb2462e780fcf14e7dc3823eb2667d0c4653112626f8c4a
    [I 10:18:20.902 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Open a web browser on your computer and enter the address

     http://127.0.0.1:8888/?token=TOKEN_FROM_OUTPUT

with the above output ```TOKEN_FROM_OUTPTU``` would bee ```ffb2462e780fcf14e7dc3823eb2667d0c4653112626f8c4a``` (this changes for every run).

    