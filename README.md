# Repository to reproduce the experiments in the paper "Deep learning observables in computational fluid dynamics"

This repository should contain everything needed to reproduce the experiments in the paper "Deep learning observables in computational fluid dynamics" by K. Lye, S. Mishra and D. Ray. (not yet published).

The following steps are needed to recreate all the plots (details about doing the ensemble runs coming).

See the ```docker``` folder for information on how to run this in Docker.

*IMPORTANT* this repository requires [git-lfs](https://git-lfs.github.com/) installed. After checking out the repository, do

    git lfs install

## Recreating the plots for single configurations

Run the script ```notebooks/submit_screen_compute_best_networks.sh``` (requires the program [GNU Screen](https://www.gnu.org/software/screen/))

## Recreating the plots for the combined configurations

Run the notebooks

  * ```notebooks/ParsingAirfoils.ipynb```
  * ```notebooks/ParsingSodShockTube.ipynb```


## Recreating the plots for the network sizes

Run the notebook ```notebooks/NetworkSizes.ipynb```
