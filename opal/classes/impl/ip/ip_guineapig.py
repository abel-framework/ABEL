import numpy as np
import tempfile
from os import remove
from string import Template
from opal import InteractionPoint, CONFIG
from opal.utilities import SI
from opal.apis.guineapig.guineapig_api import guineapig_run

class InteractionPointGUINEAPIG(InteractionPoint):
    
    # perform GUINEA-PIG simulation
    def interact(self, beam1, beam2):
        
        # make inputs
        inputs = {'energy1': beam1.energy()/1e9, # [GeV]
                  'particles1': abs(beam1.charge()/SI.e)/1e10, # [1e10]
                  'beta_x1': beam1.betaX()*1e3, # [mm]
                  'beta_y1': beam1.betaY()*1e3, # [mm]
                  'emitt_x1': beam1.normEmittanceX()*1e6, # [mm mrad]
                  'emitt_y1': beam1.normEmittanceY()*1e6, # [mm mrad]
                  'sigma_z1': beam1.bunchLength()*1e6, # [um]
                  'espread1': beam1.relEnergySpread(), # [um]
                  'offset_x1': beam1.offsetX()*1e9, # [nm]
                  'offset_y1': beam1.offsetY()*1e9, # [nm]
                  'energy2': beam2.energy()/1e9, # [GeV]
                  'particles2': abs(beam2.charge()/SI.e)/1e10, # [1e10]
                  'beta_x2': beam2.betaX()*1e3, # [mm]
                  'beta_y2': beam2.betaY()*1e3, # [mm]
                  'emitt_x2': beam2.normEmittanceX()*1e6, # [mm mrad]
                  'emitt_y2': beam2.normEmittanceY()*1e6, # [mm mrad]
                  'sigma_z2': beam2.bunchLength()*1e6, # [um]
                  'espread2': beam2.relEnergySpread(), # [um]
                  'offset_x2': beam2.offsetX()*1e9, # [nm]
                  'offset_y2': beam2.offsetY()*1e9, # [nm]
                  'ecm_min': 0.97*np.sqrt(4*beam1.energy()*beam2.energy())/1e9, # [GeV]
                  'n_x': int(np.sqrt(beam1.Npart()/2)), 
                  'n_y': int(np.sqrt(beam1.Npart()/2)), 
                  'n_z': int(np.sqrt(beam1.Npart())/3),
                  'n_t': 1,
                  'n_m1': beam1.Npart(),
                  'n_m2': beam2.Npart(),
                  'cut_x': 5*max(beam1.beamSizeX(), beam2.beamSizeX())*1e9,
                  'cut_y': 5*max(beam1.beamSizeY(), beam2.beamSizeY())*1e9,
                  'cut_z': 3*max(beam1.bunchLength(), beam2.bunchLength())*1e6}
        
        # make lattice file from template
        inputfile_template = CONFIG.opal_path + 'opal/apis/guineapig/templates/inputdeck_simple.dat'
        inputfile = tempfile.gettempdir() + '/inputdeck_simple.dat'
        with open(inputfile_template, 'r') as fin, open(inputfile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
        
        # run the simulation
        event = guineapig_run(inputfile, beam1, beam2)
        
        # delete input file
        remove(inputfile)
        
        return event
        