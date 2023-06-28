from opal import Beamline, Source, BeamDeliverySystem, Stage, Spectrometer
from matplotlib import pyplot as plt
import numpy as np

class Experiment(Beamline):
    
    def __init__(self, source=None, bds=None, stage=None, spectrometer=None):
        self.source = source
        self.bds = bds
        self.stage = stage
        self.spectrometer = spectrometer
        
        super().__init__()
    
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assemble
        assert(isinstance(self.source, Source))
        assert(isinstance(self.bds, BeamDeliverySystem))
        assert(isinstance(self.stage, Stage))
        assert(isinstance(self.spectrometer, Spectrometer))
        
        # run beamline constructor
        self.trackables = [self.source, self.bds, self.stage, self.spectrometer]
    
    
    # density plots
    def plot_spectrometer_screen(self):
        
        # load phase space
        beam = self.final_beam()

        # make screen projection
        dQdxdy, yedges, xedges = beam.density_transverse()
        
        # prepare figure
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(8)
        fig.set_figheight(5)
        
        # current profile
        c0 = ax.pcolor(xedges*1e3, yedges*1e3, abs(dQdxdy.T)*1e3, cmap='GnBu', shading='auto')
        cbar0 = fig.colorbar(c0, ax=ax)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Spectrometer screen')
        cbar0.ax.set_ylabel('Charge density (nC/mm^2)')
        
        