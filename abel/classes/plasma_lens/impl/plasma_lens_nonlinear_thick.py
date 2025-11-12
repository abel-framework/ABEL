# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel import PlasmaLens
from matplotlib import patches
import numpy as np
import scipy.constants as SI

class PlasmaLensNonlinearThick(PlasmaLens):
    
    def __init__(self, length=None, radius=None, current=None, rel_nonlinearity=0, num_slice=30):

        super().__init__(length, radius, current)
        
        # set nonlinearity (defined as R/Dx)
        self.rel_nonlinearity = rel_nonlinearity
        self.num_slice = num_slice
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # get particles
        xs  = beam.xs()-self.offset_x
        xps = beam.xps()
        ys  = beam.ys()-self.offset_y
        yps = beam.yps()

        # remove charge outside the lens (start)
        del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]

        # transport a half step
        beam.transport(self.length/(self.num_slice*2))
        
        for i in range(self.num_slice):
            

            # remove charge outside the lens (in lens)
            del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]

            # get particles
            xs  = beam.xs()-self.offset_x
            xps = beam.xps()
            ys  = beam.ys()-self.offset_y
            yps = beam.yps()

            # nominal focusing gradient
            g0 = self.get_focusing_gradient()

            # calculate the nonlinearity
            inv_Dx = self.rel_nonlinearity/self.radius

            # thin lens kick
            Bx = g0*(ys + xs*ys*inv_Dx)
            By = -g0*(xs + (xs**2 + ys**2)/2*inv_Dx)

            # calculate the angular kick
            delta_xp = self.length/self.num_slice*(By*beam.charge_sign()*SI.c/beam.Es())
            delta_yp = -self.length/self.num_slice*(Bx*beam.charge_sign()*SI.c/beam.Es())

            # set new beam positions and angles (shift back plasma-lens offsets)
            beam.set_xps(xps + delta_xp)
            beam.set_yps(yps + delta_yp)

            if i < (self.num_slice - 1): 
                # drift one step if not last step
                beam.transport(self.length/self.num_slice)
                
            
        # transport a half step
        beam.transport(self.length/(self.num_slice*2))


        # remove charge outside the lens (end)
        del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        
        return super().track(beam, savedepth, runnable, verbose)   


    def get_focusing_gradient(self):
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
    