import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick


class PlottingConfig():
    def __init__(self,name,idx,one_class=False):
        self.name=name
        self.idx=idx
        self.one_class=one_class
        self.binning = self.get_binning()
        self.logy=self.get_logy()
        self.var = self.get_name()
        self.max_y = self.get_y()


    def get_name(self):
        if 'jet' in self.name:
            name_translate = [
                r'All Jets p$_T$ [GeV]',
                r'All Jets $\eta$ [GeV]',
                r'All Jets mass [GeV]',
                r'Particle multiplicity',
            ]
        else:
            name_translate = [
                r'All particles p$_{Trel}$',
                r'All particles $\eta$',
                r'All particles $\phi$',

            ]

        return name_translate[self.idx]
    
    def get_binning(self):
        if 'jet' in self.name:
            binning_dict = {
                0 : np.linspace(1000,2500,15),
                1 : np.linspace(-2.5,2.5,10),
                2 : np.linspace(100,1500,15),
                3 : np.linspace(10,60.,10),
            }
        else:
            binning_dict = {
                0 : np.linspace(0,0.9,20),
                1 : np.linspace(-1,1,20),
                2 : np.linspace(-1,1,20),
            }
            
        return binning_dict[self.idx]

    def get_logy(self):
        if 'jet' in self.name:
            binning_dict = {
                0 : True,
                1 : False,
                2 : True,
                3 : True,
            }

        else:
            binning_dict = {
                0 : True,
                1 : True,
                2 : True,
            }
            
        return binning_dict[self.idx]



    def get_y(self):
        if 'jet' in self.name:
            binning_dict = {
                0 : 0.01,
                1 : 0.7,
                2 : 0.7,
                3 : 0.01
            }

        else:
            binning_dict = {
                0 : 30,
                1 : 2,
                2 : 2,
            }            
        return binning_dict[self.idx]
