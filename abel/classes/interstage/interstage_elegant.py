from abel.classes.interstage import Interstage
import uuid, os, shutil, subprocess, csv
import numpy as np
import scipy.constants as SI
from string import Template
from abel.CONFIG import CONFIG
from abel.apis.elegant.elegant_api import elegant_run, elegant_apl_fieldmap2D, elegant_read_beam

class InterstageElegant(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, cancel_chromaticity=True, cancel_sec_order_dispersion=True, enable_csr=True, enable_isr=True, enable_space_charge=False, num_slices=50, use_monitors=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, 
                         cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, 
                         enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)
        
        self.num_slices = num_slices
        self.use_monitors = use_monitors

    
    # track a beam through the lattice using ELEGANT
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make temporary folder and files
        parent_dir = CONFIG.temp_path
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the temporary folder
        tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
        os.mkdir(tmpfolder)
        
        # make lattice file
        filename_lattice, filename_outputbeam = self.make_elegant_lattice(beam0, tmpfolder)
        
        # make the run script
        filename_runscript, filename_inputbeam = self.make_run_script(filename_lattice, tmpfolder)

        # run ELEGANT
        from abel.apis.elegant.elegant_api import elegant_run
        beam, self.evolution = elegant_run(filename_runscript, beam0, filename_inputbeam, filename_outputbeam, verbose=verbose, runnable=runnable, tmpfolder=tmpfolder)
        #self.beam0_charge_sign = beam0.charge_sign()  # patch due to ELEGANT not setting correct charge sign

        # remove temporary files
        shutil.rmtree(tmpfolder)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def make_run_script(self, filename_lattice, tmpfolder):
        "Make the ELEGANT run script and write it to file."

        # make filenames
        filename_runscript = os.path.join(tmpfolder, 'runfile.ele')
        filename_inputbeam = os.path.join(tmpfolder, 'input_beam.bun')

        # run file template (to be filled)
        from abel.apis.elegant.elegant_api import __file__ as elegant_api_file
        filename_runfile_template = os.path.join(os.path.dirname(elegant_api_file), 'templates', 'runscript_interstage.ele')

        # template inputs
        inputs = {'nom_energy_MeV': self.nom_energy/1e6,
                  'filename_lattice': filename_lattice,
                  'filename_inputbeam': filename_inputbeam,
                  'filename_centroid': os.path.join(tmpfolder, 'centroid_vs_s.cen'),
                  'filename_twiss': os.path.join(tmpfolder, 'twiss_vs_s.twi')}

        # make new file from template
        with open(filename_runfile_template, 'r') as fin, open(filename_runscript, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)

        return filename_runscript, filename_inputbeam
    
    
    def make_elegant_lattice(self, beam, tmpfolder):
        "Make the ELEGANT lattice and write it to file."

        # make filenames
        filename_outputbeam = os.path.join(tmpfolder, 'output_beam.bun')
        
        # make lens field
        filename_lens = elegant_apl_fieldmap2D(self.nonlinearity_plasma_lens, dx=self.lens_offset_x, dy=self.lens_offset_y, tmpfolder=tmpfolder)
        
        # make lattice file from template
        from abel.apis.elegant.elegant_api import __file__ as elegant_api_file
        filename_lattice_template = os.path.join(os.path.dirname(elegant_api_file), 'templates', 'lattice_interstage.lte')

        # calculate angles
        from abel.utilities.relativity import energy2momentum
        angle_dipole = self.charge_sign*self.length_dipole*self.field_dipole*SI.e/energy2momentum(self.nom_energy)
        angle_chicane_dipole1 = self.charge_sign*self.length_chicane_dipole*self.field_chicane_dipole1*SI.e/energy2momentum(self.nom_energy)
        angle_chicane_dipole2 = self.charge_sign*self.length_chicane_dipole*self.field_chicane_dipole2*SI.e/energy2momentum(self.nom_energy)
        
        # template inputs
        
        inputs = {'charge': abs(beam.charge()),
                  'num_slices': int(self.num_slices),
                  'length_dipole': self.length_dipole,
                  'angle_dipole': angle_dipole,
                  'length_chicane_dipole': self.length_chicane_dipole,
                  'angle_chicane_dipole1': angle_chicane_dipole1,
                  'angle_chicane_dipole2': angle_chicane_dipole2,
                  'length_plasma_lens': self.length_plasma_lens,
                  'filename_lens': filename_lens,
                  'field_gradient_plasma_lens': -self.field_gradient_plasma_lens,
                  'length_gap': self.length_gap,
                  'length_central_gap_or_sextupole': self.length_central_gap_or_sextupole,
                  'strength_sextupole': self.strength_sextupole/self.length_central_gap_or_sextupole,
                  'output_filename': filename_outputbeam,
                  'enable_isr': int(self.enable_isr),
                  'enable_csr': int(self.enable_csr),
                  'monitor_filename': os.path.join(tmpfolder, 'evolution', 'output_%03ld.bun'),
                  'monitor_disabled': int(not self.use_monitors)}

        # lattice file
        filename_lattice = os.path.join(tmpfolder, 'interstage.lte')
        
        # make new file from template
        with open(filename_lattice_template, 'r') as fin, open(filename_lattice, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)

        return filename_lattice, filename_outputbeam
        
    