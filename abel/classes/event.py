import numpy as np
import openpmd_api as io
from datetime import datetime
from pytz import timezone
import scipy.constants as SI
from abel import Beam

class Event():
    
    # empty beam
    def __init__(self, input_beam1=None, input_beam2=None, shot=0):
        
        # save beams
        self.input_beam1 = input_beam1
        self.input_beam2 = input_beam2
        
        # luminosity spectrum
        self.luminosity_full = None
        self.luminosity_geom = None
        self.luminosity_peak = None
        self.upsilon_max = None
        self.num_pairs = None
        self.num_photon1 = None
        self.num_photon2 = None
        self.energy_loss1 = None
        self.energy_loss2 = None
    
    
    # calculate center of mass energy
    def center_of_mass_energy(self):
        return np.sqrt(4*self.input_beam1.energy()*self.input_beam2.energy())

    
    # luminosities
    def geometric_luminosity(self):
        return self.luminosity_geom
    
    def full_luminosity(self):
        return self.luminosity_full
    
    def peak_luminosity(self):
        return self.luminosity_peak

    def maximum_upsilon(self):
        return self.upsilon_max

    def num_coherent_pairs(self):
        return self.num_pairs
        
    def num_photons_beam1(self):
        return self.num_photon1

    def num_photons_beam2(self):
        return self.num_photon2

    def energy_loss_per_particle_beam1(self):
        return self.energy_loss1

    def energy_loss_per_particle_beam2(self):
        return self.energy_loss2

    
    # filename generator
    def filename(self, runnable, shot1=None, shot2=None):
        return runnable.shot_path(shot1, shot2) + 'event.h5'
    
    # save event (to OpenPMD format)
    def save(self, runnable, shot1=None, shot2=None):
        
        # open a new file
        series = io.Series(self.filename(runnable, shot1, shot2), io.Access.create)
        
        # add metadata
        series.author = "ABEL (the Adaptable Beginning-to-End Linac simulation framework)"
        series.date = datetime.now(timezone('CET')).strftime('%Y-%m-%d %H:%M:%S %z')
        
        # add attributes
        for key, value in self.__dict__.items():
            if not "_beam" in key:
                series.iterations[0].set_attribute(key, value)
        
        # flush
        series.flush()
        
        # save beams
        self.input_beam1.save(beam_name="input_beam1", series=series)
        self.input_beam2.save(beam_name="input_beam2", series=series)
        
        # now the file is closed
        del series
        
    
    # load event (from OpenPMD format)
    @classmethod
    def load(_, filename, load_beams=True):
        
        # create event
        event = Event()
        
        # load all beams
        if load_beams:
            event.input_beam1 = Beam.load(filename, "input_beam1")
            event.input_beam2 = Beam.load(filename, "input_beam2")
        
        # load file and add metadata
        series = io.Series(filename, io.Access.read_only)
        event.luminosity_full = series.iterations[0].get_attribute("luminosity_full")
        event.luminosity_geom = series.iterations[0].get_attribute("luminosity_geom")
        event.luminosity_peak = series.iterations[0].get_attribute("luminosity_peak")
        event.upsilon_max = series.iterations[0].get_attribute("upsilon_max")
        event.num_pairs = series.iterations[0].get_attribute("num_pairs")
        event.num_photon1 = series.iterations[0].get_attribute("num_photon1")
        event.num_photon2 = series.iterations[0].get_attribute("num_photon2")
        event.energy_loss1 = series.iterations[0].get_attribute("energy_loss1")
        event.energy_loss2 = series.iterations[0].get_attribute("energy_loss2")
        
        return event
        
    
