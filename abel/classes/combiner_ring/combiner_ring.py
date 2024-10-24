from abc import abstractmethod
from matplotlib import patches
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class CombinerRing(Trackable, CostModeled):
    
    default_exit_angle = np.pi
    
    @abstractmethod
    def __init__(self, nom_energy=None, compression_factor=None, bunch_separation_incoming=None, exit_angle=default_exit_angle):

        self.nom_energy = nom_energy
        self.compression_factor = compression_factor
        self.bunch_separation_incoming = bunch_separation_incoming
        self.exit_angle = exit_angle

        self.start_with_quarter_circle = False

    
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

    def get_bunch_separation_incoming(self):
        return self.bunch_separation * self.compression_factor

    def get_bunch_separation_outgoing(self):
        return self.bunch_separation
    
    def get_cost_breakdown(self):
        return ('Combiner ring', self.get_length() * CostModeled.cost_per_length_turnaround)
    
    def energy_usage(self):
        return 0.0
    
    def survey_object(self):
        
        thetas = np.linspace(0, 2*np.pi+self.exit_angle, 200)
        radius = self.get_bend_radius()
        
        x_points = radius*np.sin(thetas)
        y_points = radius*(1-np.cos(thetas))

        if self.start_with_quarter_circle:
            
            thetas2 = np.linspace(np.pi, 3*np.pi/2, 50)
            x_points = np.append(x_points, radius*np.sin(thetas2))
            y_points = np.append(y_points, radius*(1-np.cos(thetas2)))
            thetas3 = np.linspace(np.pi/2, 0, 50)
            x_points = np.append(x_points, radius*np.sin(thetas3)-2*radius)
            y_points = np.append(y_points, radius*(1-np.cos(thetas3))+radius*0.1)

        x_points = np.flip(x_points)
        y_points = np.flip(y_points)

        # ensure the start is at the origin
        x_points = x_points - x_points[0]
        y_points = y_points - y_points[0]
        
        
        final_angle = self.exit_angle
        label = 'Combiner ring'
        color = 'gray'
        return x_points, y_points, final_angle, label, color
    