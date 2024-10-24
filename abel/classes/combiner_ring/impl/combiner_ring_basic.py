import scipy.constants as SI
import copy
import numpy as np
from abel.classes.combiner_ring.combiner_ring import CombinerRing

class CombinerRingBasic(CombinerRing):
    
    def __init__(self, nom_energy=None, compression_factor=None, exit_angle=CombinerRing.default_exit_angle):
        
        super().__init__(nom_energy, compression_factor=None, exit_angle=exit_angle)
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # outgoing bunch separation (compressed)
        self.bunch_separation = beam0.bunch_separation/self.compression_factor
        self.num_bunches_in_train = beam0.num_bunches_in_train
        
        # compress the train
        beam = copy.deepcopy(beam0)
        beam.bunch_separation = self.bunch_separation
        
        return super().track(beam, savedepth, runnable, verbose)

    def get_bend_radius(self):
        return self.get_length()/(2*np.pi)
    
    def get_length(self):
        return self.get_bunch_separation_outgoing()*self.num_bunches_in_train*SI.c
