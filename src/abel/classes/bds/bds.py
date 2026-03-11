# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abc import abstractmethod
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class BeamDeliverySystem(Trackable, CostModeled):

    def __init__(self, num_bds=1):
        super().__init__()
        self.num_bds = num_bds
        self.name = 'BDS'
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass

    def get_cost_civil_construction(self, tunnel_diameter=None):
        return super().get_cost_civil_construction(tunnel_diameter=tunnel_diameter, tunnel_widening_factor=(1+0.88*(self.num_bds-1)))
        
    def get_cost_breakdown(self):
        return (f'{self.name} ({self.num_bds}x)', self.get_length() * CostModeled.cost_per_length_bds * self.num_bds)
    
    @abstractmethod
    def get_nom_energy(self):
        pass
    
    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = self.name
        color = 'lightgray'
        return x_points, y_points, final_angle, label, color