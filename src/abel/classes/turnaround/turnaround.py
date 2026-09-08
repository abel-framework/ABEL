# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abc import abstractmethod
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class Turnaround(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, use_semi_circle=False, start_with_semi_circle=False):

        super().__init__()
        
        self.nom_energy = nom_energy
        self.use_semi_circle = use_semi_circle
        self.start_with_semi_circle = start_with_semi_circle

        self.use_tunnel = False
        self.use_cutandcover = True

    
    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod 
    def get_length(self):
        pass

    @abstractmethod 
    def get_bend_radius(self):
        pass
    
    def get_nom_energy(self):
        return self.nom_energy 
        
    def get_cost_breakdown(self):
        return ('Turnaround', self.get_length() * CostModeled.cost_per_length_turnaround)
    
    def energy_usage(self):
        return 0.0
    
    def survey_object(self):
        
        thetas = np.linspace(0, np.pi, 100)
        radius = self.get_bend_radius()
        
        x_points = radius*np.sin(thetas)
        y_points = radius*(1-np.cos(thetas))
        if not self.use_semi_circle:
            thetas2 = np.linspace(np.pi, 3*np.pi/2, 50)
            x_points = np.append(x_points, radius*np.sin(thetas2))
            y_points = np.append(y_points, radius*(1-np.cos(thetas2)))
            thetas3 = np.linspace(np.pi/2, 0, 50)
            x_points = np.append(x_points, radius*np.sin(thetas3)-2*radius)
            y_points = np.append(y_points, radius*(1-np.cos(thetas3))+radius*0.1)

        if not self.start_with_semi_circle:
            x_points = np.flip(x_points)
            y_points = np.flip(y_points)

        # ensure the start is at the origin
        x_points = x_points - x_points[0]
        y_points = y_points - y_points[0]
        
        
        final_angle = np.pi 
        label = 'Turnaround'
        color = 'gray'
        return x_points, y_points, final_angle, label, color
    