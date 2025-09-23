# Copyright 2022-, The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

from abc import abstractmethod
from matplotlib import patches
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class TransferLine(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, length=None):

        super().__init__()
        
        self.nom_energy = nom_energy
        self.length = length

        self.name = 'Transfer line'

    
    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        return self.length

    def get_nom_energy(self):
        return self.nom_energy 
        
    def get_cost_breakdown(self):
        return (self.name, self.get_length() * CostModeled.cost_per_length_transfer_line)
    
    def energy_usage(self):
        return 0.0
    
    def survey_object(self):
        
        # ensure the start is at the origin
        x_points = np.array([0, self.get_length()])
        y_points = np.array([0, 0])
        
        final_angle = 0
        label = 'Transfer line'
        color = 'gray'
        return x_points, y_points, final_angle, label, color
    