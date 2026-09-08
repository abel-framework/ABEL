# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.bds.impl.plasma_lens import BeamDeliverySystemPlasmaLens
import scipy.constants as SI
import numpy as np
import copy

class BeamDeliverySystemPlasmaLensBasic(BeamDeliverySystemPlasmaLens):
    
    def __init__(self, nom_energy=None, beta0=None, alpha0=0, beta_star=None, L_star=2.0, length_scale=2.0, Dpx_star=0.04, charge_sign=-1, cancel_chromaticity=True, enable_csr=True, enable_isr=True):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, alpha0=alpha0, beta_star=beta_star, L_star=L_star, length_scale=length_scale, Dpx_star=Dpx_star, charge_sign=charge_sign, cancel_chromaticity=cancel_chromaticity, enable_csr=enable_csr, enable_isr=enable_isr)
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # transport phase spaces to waist (in each plane)
        ds_x = beam.alpha_x()/beam.gamma_x()
        ds_y = beam.alpha_y()/beam.gamma_y()
        
        # find waist beta functions (in each plane)
        Rx = np.eye(4)
        Rx[0,1] = ds_x
        Rx[2,3] = ds_x
        beamx = copy.deepcopy(beam)
        beamx.set_transverse_vector(np.dot(Rx, beamx.transverse_vector()))

        Ry = np.eye(4)
        Ry[0,1] = ds_y
        Ry[2,3] = ds_y
        beamy = copy.deepcopy(beam)
        beamy.set_transverse_vector(np.dot(Ry, beamy.transverse_vector()))
        
        # scale the waist phase space by beta functions
        X = beamx.transverse_vector()
        Y = beamy.transverse_vector()
        X[0,:] = X[0,:] * np.sqrt(self.beta_star/beamx.beta_x())
        X[1,:] = X[1,:] / np.sqrt(self.beta_star/beamx.beta_x())
        X[2,:] = Y[2,:] * np.sqrt(self.beta_star/beamy.beta_y())
        X[3,:] = Y[3,:] / np.sqrt(self.beta_star/beamy.beta_y()) 
        beam.set_transverse_vector(X)
        
        return super().track(beam, savedepth, runnable, verbose)
        