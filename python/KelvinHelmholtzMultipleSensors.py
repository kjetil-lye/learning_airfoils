
# coding: utf-8

# # Machine learning training for KH
# *NOTE* You need to set the environment variable "AIRFOILS_DLMC_KH_DATAPATH" to the folder containing "kh_1.nc" and "qmc_points.txt"

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('../python')
from machine_learning import *
import os


def get_kh_data():

    variables = ['rho']

    points = [[0.55,0.35], [0.75,0.75]]
    func_names=["$Q_2$", "$Q_1$"]
    if 'AIRFOILS_DLMC_KH_DATAPATH' not in os.environ:
        print("The environment varialbe AIRFOILS_DLMC_KH_DATAPATH should point to a folder containing 'kh_1.nc' and 'qmc_points.txt'")
        sys.exit(1)
        
    data_path=os.path.join(os.environ['AIRFOILS_DLMC_KH_DATAPATH'], 'kh_1.nc')
    if not os.path.exists(data_path):
        print("The environment varialbe AIRFOILS_DLMC_KH_DATAPATH should point to a folder containing 'kh_1.nc' and 'qmc_points.txt'")
        sys.exit(1)
        
    parameter_path = os.path.join(os.path.dirname(data_path), 'qmc_points.txt')

    if not os.path.exists(parameter_path):
        print("The environment varialbe AIRFOILS_DLMC_KH_DATAPATH should point to a folder containing 'kh_1.nc' and 'qmc_points.txt'")
        sys.exit(1)

    parameters = np.loadtxt(parameter_path)


    input_size=40
    train_size=128
    validation_size=128



    data_per_func = {}
    for (func_name, p) in zip(func_names, points):
        
        functional =  AreaFunctional(integrate_coordinate=p,
                                     variable=v,
                                     short_title=func_name)
                                
        samples = get_samples(data_path, functional)

        data_per_func[func_name] = samples

    return parameters, data_per_func, parameters, data_per_func

def get_kh_network():
    kh_network = [20, 22, 20, 22, 20, 20, 1]

    return kh_network


class AreaFunctional(object):
    def __init__(self, integrate_coordinate = [0.55,0.35], integrate_width=[0.25,0.25],variable='rho', 
                 short_title=None):
        self.integrate_coordinate = integrate_coordinate
        self.integrate_width = integrate_width
        
        self.variable = variable
        self.first = 10
        
        if short_title is None:
            self.__short_title = self.title()
        else:
            self.__short_title = short_title
        
    def short_title(self):
        return self.__short_title
        
    def title(self):
        return 'integrated area: $[%.2f,%.2f]\\times [%.2f,%.2f]$'% (self.integrate_coordinate[0],
                                                            self.integrate_coordinate[0]+self.integrate_width[0],
                                                            self.integrate_coordinate[1],
                                                            self.integrate_coordinate[1]+self.integrate_width[1])
    def area(self, I):
        
        return 1*(I[1][0]-I[0][0])*(I[1][1]-I[0][1])
    def __call__(self, rho):
        N = rho.shape[0]
        
        integrate_area = [[int(N*self.integrate_coordinate[0]), int(N*self.integrate_coordinate[1])],
                          [int(N*self.integrate_coordinate[0]+N*self.integrate_width[0]), 
                           int(N*self.integrate_coordinate[1]+N*self.integrate_width[1])]]
        
        g = np.sum(rho[integrate_area[0][0]:integrate_area[1][0], integrate_area[0][1]:integrate_area[1][1]])/self.area(integrate_area)
        
      
        return g
    
    def plot(self, d):
        N = d.shape[0]
        x, y= mgrid[0:1:N*1j,0:1:N*1j]
        plt.pcolormesh(x,y,d)
        rect = matplotlib.patches.Rectangle((self.integrate_coordinate[0],self.integrate_coordinate[1]),
                                     self.integrate_width[0],
                                     self.integrate_width[1],
                                     linewidth=1,edgecolor='r',facecolor='none')
        
        axes = plt.gca()
        axes.add_patch(rect)
        
        
        
class SinglePointFunctional(object):
    def __init__(self, coordinate = [0.55,0.35], variable='rho'):
        self.coordinate = coordinate
        
        self.variable = variable
        self.first = 10
         
    def short_title(self):
        return '$(%f,%f)$'% (self.coordinate[0], self.coordinate[1])
    def title(self):
        return 'specific point  $(%f,%f)$'% (self.coordinate[0], self.coordinate[1])

    def __call__(self, rho):
        N = rho.shape[0]
        
        x = int(N*self.coordinate[0])
        y = int(N*self.coordinate[1])
        
        return rho[x,y]
    
    def plot(self, d):
        N = d.shape[0]
        x, y= mgrid[0:1:N*1j,0:1:N*1j]
        plt.pcolormesh(x,y,d)
        rect = matplotlib.patches.Rectangle((self.coordinate[0],self.coordinate[1]),
                                     0.05,
                                     0.05,
                                     linewidth=1,edgecolor='r',facecolor='r')
        
        axes = plt.gca()
        axes.add_patch(rect)
        
    
    


# In[4]:


def get_samples(data_path, functional):
    samples = []
    
    with netCDF4.Dataset(data_path) as f:
        for k in f.variables.keys():
            if functional.variable in k:
                sys.stdout.write("%d\r" % len(samples))
                sys.stdout.flush()
                samples.append(functional(f.variables[k][:,:,0]))
                
               
    print()
    return array(samples,dtype=float64)
def draw_functional(data_path, functional):
    with netCDF4.Dataset(data_path) as f:
        d = f.variables['sample_10_%s' % functional.variable][:,:,0]
        functional.plot(d)
        

if __name__ == '__main__':
    kh_network = get_kh_network()


    parameters, data_per_func,_,_ = get_kh_data()


    for force_name in data_per_func.keys():
        try_best_network_sizes(parameters=parameters,
                               samples=data_per_func[force_name],
                               base_title='KH %s' % force_name)




    for force_name in data_per_func.keys():
        train_single_network(parameters=parameters,
                             samples=data_per_func[force_name],
                             base_title='KH %s' % force_name,
                             network = airfoils_network)


