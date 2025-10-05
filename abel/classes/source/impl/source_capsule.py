# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.source.source import Source

class SourceCapsule(Source):
    """
    Source class used for "encapsulating" a predefined beam to be passed into 
    e.g. a ``Stage``. used e.g. between ramps in plasma stages.

    Attributes
    ----------
    beam : ``Beam``
        Beam to be encapsulated.
    """
    
    def __init__(self, length=0, beam=None, energy=None, charge=None, x_offset=0, y_offset=0, x_angle=0, y_angle=0, wallplug_efficiency=1, accel_gradient=None):
        
        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle)

        self.beam = beam

    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        """
        Return a deep copy of ``self.beam``.
        """

        import copy
        
        # return the saved beam
        beam = copy.deepcopy(self.beam)
        
        # add jitters and offsets in super function
        return beam
        #return super().track(beam, savedepth, runnable, verbose)
    