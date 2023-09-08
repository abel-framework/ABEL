import numpy as np
import openpmd_api as io
from datetime import datetime
from pytz import timezone
import scipy.constants as SI
from abel import Beam

class Event():
    
    # empty beam
    def __init__(self, input_beam1=None, input_beam2=None):
        
        # save beams
        self.input_beam1 = input_beam1
        self.input_beam2 = input_beam2
        
        # luminosity spectrum
        self.luminosity_full = None
        self.luminosity_geom = None
        self.luminosity_peak = None
    
    
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
    
    # filename generator
    def filename(self, runnable):
        return runnable.shot_path() + 'event.h5'
    
    # save event (to OpenPMD format)
    def save(self, runnable):
        
        # open a new file
        series = io.Series(self.filename(runnable), io.Access.create)
        
        # add metadata
        series.author = "ABEL (the Advanced Beams and Extreme Linacs code)"
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
        
        return event
        
    