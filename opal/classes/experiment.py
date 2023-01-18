from opal import Runnable, Beam, Beamline, Source, BeamDeliverySystem, Stage, Spectrometer
from matplotlib import pyplot as plt
import numpy as np

class Experiment(Beamline):
    
    # constructor
    def __init__(self, source=None, bds=None, stage=None, spectrometer=None):
        
        # check element classes, then assemble
        assert(isinstance(source, Source))
        assert(isinstance(bds, BeamDeliverySystem))
        assert(isinstance(stage, Stage))
        assert(isinstance(spectrometer, Spectrometer))
        
        # run linac constructor
        trackables = [source, bds, stage, spectrometer]
        super().__init__(trackables)
    
    
    # density plots
    def plotSpectrometerScreen(self):
        
        # load phase space
        files = self.runData()
        beam = Beam.load(files[-1][-1])

        # make screen projection
        dQdxdy, yedges, xedges = beam.densityTransverse()
        
        # prepare figure
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(8)
        fig.set_figheight(5)
        
        # current profile
        c0 = ax.pcolor(xedges*1e3, yedges*1e3, abs(dQdxdy.T)*1e6, cmap='GnBu')
        cbar0 = fig.colorbar(c0, ax=ax)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Spectrometer screen')
        cbar0.ax.set_ylabel('Charge density (pC/mm^2)')
        
        