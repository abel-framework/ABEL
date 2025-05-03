from abc import abstractmethod
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class DampingRing(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, emit_nx_target=None, emit_ny_target=None, bunch_separation_in_ring=None, num_rings=1):

        super().__init__()
        
        self.nom_energy = nom_energy
        self.emit_nx_target = emit_nx_target
        self.emit_ny_target = emit_ny_target
        
        self.bunch_separation_in_ring = bunch_separation_in_ring
        self.num_rings = num_rings

        self.name = 'Damping ring'
      
    
    
    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod 
    def get_length(self):
        pass
    
    def get_nom_energy(self):
        return self.nom_energy 

    @abstractmethod 
    def get_damping_time(self):
        pass

    @abstractmethod 
    def get_circumference(self): # not the same as get_length
        pass

    
    def get_cost_breakdown(self):
        breakdown = []
        breakdown.append(('Ring components', self.get_circumference() * CostModeled.cost_per_length_damping_ring))
        breakdown.append(('Civil construction', self.get_circumference() * CostModeled.cost_per_length_tunnel))
        return (self.name, breakdown)

    
    @abstractmethod 
    def energy_usage(self):
        pass
    
    def survey_object(self):
        
        thetas = np.linspace(0, 2*np.pi, 200)
        radius = self.get_circumference()/(2*np.pi)
        x_points = radius*np.sin(thetas)
        y_points = -radius*(1-np.cos(thetas))
            
        final_angle = 0
        label = self.name
        color = 'green'
        return x_points, y_points, final_angle, label, color
        
    