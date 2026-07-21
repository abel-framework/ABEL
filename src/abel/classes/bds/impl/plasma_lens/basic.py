# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.bds.impl.plasma_lens import BeamDeliverySystemPlasmaLens
import scipy.constants as SI
import numpy as np

class BeamDeliverySystemPlasmaLensBasic(BeamDeliverySystemPlasmaLens):
    
    def __init__(self, nom_energy=None, beta0=None, alpha0=0, beta_star=None, L_star=2.0, field_final_dipole=None):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, alpha0=alpha0, beta_star=beta_star, L_star=L_star, field_final_dipole=field_final_dipole)
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        # does nothing for now
        return super().track(beam, savedepth, runnable, verbose)
        