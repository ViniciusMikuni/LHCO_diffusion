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
                r'All Jets $\eta$',
                r'All Jets $\phi$',
                r'All Jets mass [GeV]',
                r'Particle multiplicity',
            ]

        elif 'High' in self.name:
            name_translate = [
                r'All Jets m [GeV]',
                r'All Jets $\tau_1$',
                r'All Jets $\tau_2$',
                r'All Jets $\tau_3$',
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
                0 : np.linspace(1000,2500,35),
                1 : np.linspace(-2.5,2.5,30),
                2 : np.linspace(-1.5,1.5,30),
                3 : np.linspace(100,1100,35),
                4 : np.linspace(1,100.,100),
            }
        elif 'High' in self.name:
            binning_dict = {
                0 : np.linspace(100,550,35),
                1 : np.linspace(0,600.0,30),
                2 : np.linspace(0,500.0,30),
                3 : np.linspace(0,400.0,30),
            }
        else:
            binning_dict = {
                0 : np.linspace(0,0.9,40),
                1 : np.linspace(-1,1,30),
                2 : np.linspace(-1,1,30),
            }
            
        return binning_dict[self.idx]

    def get_logy(self):
        if 'jet' in self.name:
            binning_dict = {
                0 : True,
                1 : False,
                2 : False,
                3 : True,
                4 : True,
            }
        elif 'High' in self.name:
            binning_dict = {
                0 : True,
                1 : True,
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
                1 : 0.9,
                2 : 0.7,
                3 : 0.01,
                4 : 0.1
            }
            
        elif 'High' in self.name:
            binning_dict = {
                0 : 0.01,
                1 : 0.1,
                2 : 0.1,
                3 : 0.1,
            }

        else:
            binning_dict = {
                0 : 30,
                1 : 5,
                2 : 5,
            }            
        return binning_dict[self.idx]
