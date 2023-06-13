import tempfile, os
import numpy as np
import scipy.constants as SI
from string import Template
from opal import InteractionPoint, CONFIG
from opal.apis.guineapig.guineapig_api import guineapig_run

class InteractionPointGuineaPig(InteractionPoint):
    
    def __init__(self, enable_waist_shift = False):
        self.enable_waist_shift = enable_waist_shift
        
    # perform GUINEA-PIG simulation
    def interact(self, beam1, beam2, load_beams=False):
        
        # make inputs
        inputs = {'energy1': beam1.energy()/1e9, # [GeV]
                  'particles1': abs(beam1.charge()/SI.e)/1e10, # [1e10]
                  'beta_x1': beam1.beta_x()*1e3, # [mm]
                  'beta_y1': beam1.beta_y()*1e3, # [mm]
                  'emitt_x1': beam1.norm_emittance_x()*1e6, # [mm mrad]
                  'emitt_y1': beam1.norm_emittance_y()*1e6, # [mm mrad]
                  'sigma_z1': beam1.bunch_length()*1e6, # [um]
                  'espread1': beam1.rel_energy_spread(),
                  'offset_x1': beam1.x_offset()*1e9, # [nm]
                  'offset_y1': beam1.y_offset()*1e9, # [nm]
                  'waist_x1': 0, # [um]
                  'waist_y1': int(self.enable_waist_shift)*beam1.bunch_length()*1e6, # [um]
                  'energy2': beam2.energy()/1e9, # [GeV]
                  'particles2': abs(beam2.charge()/SI.e)/1e10, # [1e10]
                  'beta_x2': beam2.beta_x()*1e3, # [mm]
                  'beta_y2': beam2.beta_y()*1e3, # [mm]
                  'emitt_x2': beam2.norm_emittance_x()*1e6, # [mm mrad]
                  'emitt_y2': beam2.norm_emittance_y()*1e6, # [mm mrad]
                  'sigma_z2': beam2.bunch_length()*1e6, # [um]
                  'espread2': beam2.rel_energy_spread(), # [um]
                  'offset_x2': beam2.x_offset()*1e9, # [nm]
                  'offset_y2': beam2.y_offset()*1e9, # [nm]
                  'waist_x2': 0, # [um]
                  'waist_y2': int(self.enable_waist_shift)*beam2.bunch_length()*1e6, # [um]
                  'ecm_min': 0.99*np.sqrt(4*beam1.energy()*beam2.energy())/1e9, # [GeV] 1% peak
                  'n_x': int(np.sqrt(len(beam1)/2)),
                  'n_y': int(np.sqrt(len(beam1)/2)),
                  'n_z': int(np.sqrt(len(beam1)/2)),
                  'n_t': 10,
                  'n_m1': len(beam1),
                  'n_m2': len(beam2),
                  'cut_x': max(abs(beam1.x_offset())+4*beam1.beam_size_x(), abs(beam2.x_offset())+4*beam2.beam_size_x())*1e9,
                  'cut_y': max(abs(beam1.y_offset())+4*beam1.beam_size_y(), abs(beam2.y_offset())+4*beam2.beam_size_y())*1e9,
                  'cut_z': 4*max(beam1.bunch_length(), beam2.bunch_length())*1e6,
                  'load_beam': 3*int(load_beams)}
        
        # make lattice file from template
        inputfile_template = CONFIG.opal_path + 'opal/apis/guineapig/templates/inputdeck_simple.dat'
        inputfile = tempfile.gettempdir() + '/inputdeck_simple.dat'
        with open(inputfile_template, 'r') as fin, open(inputfile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
        
        # run the simulation
        event = guineapig_run(inputfile, beam1, beam2)
        
        # delete input file
        os.remove(inputfile)
        
        return event
        