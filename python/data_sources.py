import SodShockTube
import SodShockTubeQMC
import MachineLearningSixParametersAirfoil
import GaussianRandomVariable
data_sources = {
    'Airfoils' : [
        MachineLearningSixParametersAirfoil.get_airfoil_data,
        MachineLearningSixParametersAirfoil.get_airfoils_network
    ],

    'SodShockTubeQMC' : [
        SodShockTubeQMC.get_sod_data_qmc,
        SodShockTubeQMC.get_network
    ],

    'Sine' : [
        GaussianRandomVariable.get_sine_data,
        GaussianRandomVariable.get_sine_network
    ]
}
