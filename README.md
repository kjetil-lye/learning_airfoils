# Repository to reproduce the experiments in the paper "Deep learning observables in computational fluid dynamics"

This repository should contain everything needed to reproduce the experiments in the paper ["Deep learning observables in computational fluid dynamics" by K. Lye, S. Mishra and D. Ray. (preprint, arXiv:1903.03040)](https://arxiv.org/abs/1903.03040).

The following steps are needed to recreate all the plots (details about doing the ensemble runs coming).

See the ```docker``` folder for information on how to run this in Docker.

*IMPORTANT* this repository requires [git-lfs](https://git-lfs.github.com/) installed. Make sure you have this package installed (```git lfs install``` is **not** enough)

After checking out the repository, and after installing git-lfs, do

    git lfs install
    git lfs pull

in the repository directory. This will fetch all the additional data files (except the KH data files -- these are not needed for any of the currently described runs).

## Running these scripts

Every script can be run from a BASH compatible command line, or through the
supplied Docker container (documented under ```docker```).

## Recreating the tables for the network size analysis

Run the script file ```notebooks/network_size_analysis.sh``` from the folder ```notebooks```

## Recreating the plots for single configurations

Run the script ```notebooks/run_best_and_effective_networks.sh``` from the folder ```notebooks```

## Recreating the plots for the combined configurations

Run the notebooks

  * ```notebooks/ParsingAirfoils.ipynb```
  * ```notebooks/ParsingSodShockTube.ipynb```


## Recreating the plots for the network sizes

Run the notebook ```notebooks/NetworkSizes.ipynb```
