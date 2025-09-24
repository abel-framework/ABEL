# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.plasma_lens.plasma_lens import PlasmaLens
import numpy as np
import scipy.constants as SI

class PlasmaLensNonlinearThin(PlasmaLens):
    
    def __init__(self, length=None, radius=None, current=None, rel_nonlinearity=0):

        super().__init__(length, radius, current)
        
        # set nonlinearity (defined as R/Dx)
        self.rel_nonlinearity = rel_nonlinearity
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # remove charge outside the lens (start)
        del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        
        # drift half the distance
        beam.transport(self.length/2)
        
        # remove charge outside the lens (middle)
        del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        
        # get particles
        xs = beam.xs()-self.offset_x
        xps = beam.xps()
        ys = beam.ys()-self.offset_y
        yps = beam.yps()

        # nominal focusing gradient
        g0 = self.get_focusing_gradient()

        # calculate the nonlinearity
        inv_Dx = self.rel_nonlinearity/self.radius
        
        # thin lens kick
        Bx = g0*(ys + xs*ys*inv_Dx)
        By = -g0*(xs + (xs**2 + ys**2)/2*inv_Dx)

        # calculate the angular kicks
        delta_xp = self.length*(By*beam.charge_sign()*SI.c/beam.Es())
        delta_yp = -self.length*(Bx*beam.charge_sign()*SI.c/beam.Es())
        
        # set new beam positions and angles (shift back plasma-lens offsets)
        beam.set_xps(xps + delta_xp)
        beam.set_yps(yps + delta_yp)

        # drift another half the distance
        beam.transport(self.length/2)

        # remove charge outside the lens (end)
        del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        
        return super().track(beam, savedepth, runnable, verbose)   


    def get_focusing_gradient(self):
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
    