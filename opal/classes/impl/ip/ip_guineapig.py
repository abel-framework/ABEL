import numpy as np
import tempfile
from os import remove
from string import Template
from opal import InteractionPoint, CONFIG
from opal.utilities import SI
from opal.apis.guineapig.guineapig_api import guineapig_run

class InteractionPointGUINEAPIG(InteractionPoint):
    
    def __init__(self, enableWaistShift = False):
        self.enableWaistShift = enableWaistShift
        
    # perform GUINEA-PIG simulation
    def interact(self, beam1, beam2, loadBeams=False):
        
        # make inputs
        inputs = {'energy1': beam1.energy()/1e9, # [GeV]
                  'particles1': abs(beam1.charge()/SI.e)/1e10, # [1e10]
                  'beta_x1': beam1.betaX()*1e3, # [mm]
                  'beta_y1': beam1.betaY()*1e3, # [mm]
                  'emitt_x1': beam1.normEmittanceX()*1e6, # [mm mrad]
                  'emitt_y1': beam1.normEmittanceY()*1e6, # [mm mrad]
                  'sigma_z1': beam1.bunchLength()*1e6, # [um]
                  'espread1': beam1.relEnergySpread(),
                  'offset_x1': beam1.offsetX()*1e9, # [nm]
                  'offset_y1': beam1.offsetY()*1e9, # [nm]
                  'waist_x1': 0, # [um]
                  'waist_y1': int(self.enableWaistShift)*beam1.bunchLength()*1e6, # [um]
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
                  'waist_x2': 0, # [um]
                  'waist_y2': int(self.enableWaistShift)*beam2.bunchLength()*1e6, # [um]
                  'ecm_min': 0.99*np.sqrt(4*beam1.energy()*beam2.energy())/1e9, # [GeV] 1% peak
                  'n_x': int(np.sqrt(beam1.Npart()/2)),
                  'n_y': int(np.sqrt(beam1.Npart()/2)),
                  'n_z': int(np.sqrt(beam1.Npart()/2)),
                  'n_t': 10,
                  'n_m1': beam1.Npart(),
                  'n_m2': beam2.Npart(),
                  'cut_x': max(abs(beam1.offsetX())+4*beam1.beamSizeX(), abs(beam2.offsetX())+4*beam2.beamSizeX())*1e9,
                  'cut_y': max(abs(beam1.offsetY())+4*beam1.beamSizeY(), abs(beam2.offsetY())+4*beam2.beamSizeY())*1e9,
                  'cut_z': 4*max(beam1.bunchLength(), beam2.bunchLength())*1e6,
                  'load_beam': 3*int(loadBeams)}
        
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
        