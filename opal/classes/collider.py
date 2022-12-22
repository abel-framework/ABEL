from opal import Runnable, Linac, BeamDeliverySystem, InteractionPoint
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from copy import deepcopy

class Collider(Runnable):
    
    # constructor
    def __init__(self, linac1=None, bds1=None, ip=None, linac2=None, bds2=None):
        
        # check element classes, then assemble
        assert(isinstance(linac1, Linac))
        assert(isinstance(bds1, BeamDeliverySystem))
        assert(isinstance(ip, InteractionPoint))
        
        self.linac1 = linac1
        self.linac2 = linac2
        self.bds1 = bds1
        self.bds2 = bds2
        self.ip = ip
    
    
    # run simulation
    def run(self, runname=None, shots=1, savedepth=2, verbose=True, overwrite=True):
        
        # copy second arm if undefined
        if self.linac2 is None:
            self.linac2 = deepcopy(self.linac1)
        if self.bds2 is None:
            self.bds2 = deepcopy(self.bds1)
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "collider_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
        
        # declare shots list
        self.shots = []
        
        # make base folder and clear tracking directory
        if not exists(self.runPath()):
            mkdir(self.runPath())
        
        # run first linac arm
        if verbose:
            print(">> LINAC #1")
        beam1 = self.linac1.run(self.runname + "/linac1", shots, savedepth, verbose, overwrite)
        
        # run second linac arm
        if verbose:
            print(">> LINAC #2")
        beam2 = self.linac2.run(self.runname + "/linac2", shots, savedepth, verbose, overwrite)
        
        if verbose:
            print(">> INTERACTION POINT")
        event = self.ip.run(self.linac1, self.linac2, self.runname + "/ip", allByAll=False)
        
        # return beams from last shot
        return beam1, beam2, event
    
    
    # TODO: some plots with the luminosity etc.
    