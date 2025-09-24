# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.source.source import Source
import os

class SourceFromFile(Source):
    
    def __init__(self, length=0, charge=None, energy=None, accel_gradient=None, wallplug_efficiency=1, file=None, x_offset=0, y_offset=0, x_angle=0, y_angle=0, waist_shift_x=0, waist_shift_y=0):

        if file is not None and not os.path.exists(file):
            raise FileNotFoundError(f"Error: The file '{file}' was not found.")
        
        self._file = file

        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle, waist_shift_x, waist_shift_y)

    
    @property
    def file(self) -> str | None:
        return self._file
    @file.setter
    def file(self, file_path : str):
        if file_path is not None and not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
        else:
            self._file = file_path
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):

        from abel.classes.beam import Beam
        
        # make empty beam
        beam = Beam.load(self.file)

        # scale the charge (if set)
        if self.charge is not None:
            beam.scale_charge(self.charge)
        else:
            self.charge = beam.charge()

        # scale the energy (if set)
        if self.energy is not None:
            beam.scale_energy(self.energy)
        else:
            self.energy = beam.energy()

        # add jitters and offsets in super function
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_charge(self):
        if self.charge is None:
            beam = Beam.load(self.file)
            self.charge = beam.charge()
        return self.charge

    
    def get_energy(self):
        if self.energy is None:
            beam = Beam.load(self.file)
            self.energy = beam.energy()
        return self.energy
    