# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target.sfqed import TargetSFQED
import os, uuid
from abel.CONFIG import CONFIG

class TargetSFQEDPtarmigan(TargetSFQED):
    
    def __init__(self, laser_a0=None, laser_waist_size=None, laser_duration=None, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0):
        
        super().__init__(laser_a0=laser_a0, laser_waist_size=laser_waist_size, laser_duration=laser_duration, laser_wavelength=laser_wavelength, laser_polarization=laser_polarization, collision_angle_deg=collision_angle_deg)

    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        from abel.wrappers.ptarmigan.ptarmigan_wrapper import ptarmigan_write_inputs, ptarmigan_run

        ## PREPARE TEMPORARY FOLDER
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.makedirs(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + os.sep
        
        # make directory
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
            
        # define input file
        filename_input = 'ptarmigan.yml'
        path_input = os.path.join(tmpfolder, filename_input)
        
        # make input file
        ptarmigan_write_inputs(path_input, beam, self.laser_a0, self.laser_wavelength, self.laser_duration, self.laser_waist_size, self.laser_polarization, self.collision_angle_deg)

        # perform ptarmigan simulation
        ptarmigan_run(path_input)

        # TODO: extract the information from the H5 file
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def peak_chi(self):
        return None