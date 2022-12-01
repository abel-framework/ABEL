from opal import Beam, Trackable, Linac, Source, BeamDeliverySystem, Stage, Spectrometer
from matplotlib import pyplot as plt
#from os.path import isfile, join, exists
#from os import listdir, remove, mkdir
import numpy as np

class LinacExperiment(Linac):
    
    # constructor
    def __init__(self, source=None, bds=None, stage=None, spectrometer=None, runname=None):
        
        # check element classes, then assemble
        assert(isinstance(source, Source))
        assert(isinstance(bds, BeamDeliverySystem))
        assert(isinstance(stage, Stage))
        assert(isinstance(spectrometer, Spectrometer))
        
        # run linac constructor
        trackables = [source, bds, stage, spectrometer]
        super().__init__(trackables, runname)
    
    
    # density plots
    def plotSpectrometer(self):
        
        # load phase space
        files = self.trackData()
        beam = Beam.load(self.trackPath() + files[-1])

        # make screen projection
        image, xedges, yedges = np.histogram2d(beam.xs(), beam.ys())
        
        # prepare figure
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(4)
        fig.set_figheight(10)
        
        # current profile
        c0 = ax.pcolor(yedges*1e6, xedges*1e6, image, cmap='GnBu')
        cbar0 = fig.colorbar(c0, ax=ax)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        cbar0.ax.set_ylabel('Charge density (counts)')
        
        