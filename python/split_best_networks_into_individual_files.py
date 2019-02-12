"""
Simple script that splits the file data/best_networks.json into
one json file per network
"""

import os
import json
def split_best_networks_into_individual_files(best_network_file, outputdir):
    with open(best_network_file) as infile:
        best_networks = json.load(infile)

        for best_network_name in best_networks.keys():
            with open(os.path.join(outputdir, best_network_name + "_best_network.json"), 'w') as outfile:
                json.dump({best_network_name : best_networks[best_network_name]}, outfile, indent=4)


if __name__=='__main__':
    datafolder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    split_best_networks_into_individual_files(os.path.join(datafolder, 'best_networks.json'),
                                              datafolder)
