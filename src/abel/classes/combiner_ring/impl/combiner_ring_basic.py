# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import scipy.constants as SI
import numpy as np
from abel.classes.combiner_ring.combiner_ring import CombinerRing

class CombinerRingBasic(CombinerRing):
    
    def __init__(self, nom_energy=None, compression_factor=None, exit_angle=CombinerRing.default_exit_angle):
        
        super().__init__(nom_energy=nom_energy, compression_factor=None, exit_angle=exit_angle)

        self.max_dipole_field = 1.5
        self.max_rel_energy_loss = 0.01

        self.suppress_unphysical_warning = False
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        import copy
        
        # outgoing bunch separation (compressed)
        self.bunch_separation = beam0.bunch_separation/self.compression_factor
        self.num_bunches_in_train = beam0.num_bunches_in_train
        
        # compress the train
        beam = copy.deepcopy(beam0)
        beam.bunch_separation = self.bunch_separation

        # warn if field or energy loss is too high
        for i in range(10):
            self.ring_fill_factor = 1.0/(i+1)
            
            if not self.nom_energy is None:
                if self.get_average_dipole_field() > self.max_dipole_field:
                    if self.suppress_unphysical_warning:
                        continue
                    else:
                        print(f"Dipole field too high: {self.get_average_dipole_field():.1f} T")
                if self.get_rel_energy_loss() > self.max_rel_energy_loss:
                    if self.suppress_unphysical_warning:
                        continue
                    else:
                        print(f"Relative energy loss too high: {100*self.get_rel_energy_loss():.1f}%")
            else:
                print("CombinerRingBasic: Could not evaluate dipole field or energy loss limits because nom_energy not set")
            
        return super().track(beam, savedepth, runnable, verbose)

    def get_bend_radius(self):
        return self.get_length()/(2*np.pi)
    
    def get_length(self):
        return self.get_bunch_separation_outgoing()*self.num_bunches_in_train*SI.c / self.ring_fill_factor

    def get_average_dipole_field(self):
        # average dipole field based on energy and radius
        return self.nom_energy / (SI.c * self.get_bend_radius())
    
    def get_num_turns(self):
        return self.compression_factor
        
    def get_rel_energy_loss(self):
        
        rel_energy_loss_per_turn = 2 * np.pi * SI.e**2 * (self.nom_energy*SI.e)**3 / (6 * np.pi * SI.epsilon_0 * self.get_bend_radius() * (SI.m_e*SI.c**2)**4)

        return self.get_num_turns() * rel_energy_loss_per_turn
        
