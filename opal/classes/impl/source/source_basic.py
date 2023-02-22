from opal import Source, Beam
from opal.utilities import SI
from opal.utilities.beamphysics import generateTraceSpace
from opal.utilities.relativity import energy2gamma
import numpy as np
from types import SimpleNamespace
import time

class SourceBasic(Source):
    
    def __init__(self, E0 = None, Q = None, sigE = None, relsigE = None, sigz = None, z = 0, x0 = 0, y0 = 0, emitnx = None, emitny = None, betax = None, betay = None, alphax = 0, alphay = 0, L = 0, Npart = 1000, wallplugEfficiency = 1, acceleratingGradient = None):
        self.E0 = E0
        self.Q = Q
        self.sigE = sigE # [eV]
        self.relsigE = relsigE
        self.sigz = sigz # [m]
        self.z = z # [m]
        self.x0 = x0 # [m]
        self.y0 = y0 # [m]
        self.Npart = Npart
        self.emitnx = emitnx # [m rad]
        self.emitny = emitny # [m rad]
        self.betax = betax # [m]
        self.betay = betay # [m]
        self.alphax = alphax # [m]
        self.alphay = alphay # [m]
        self.L = L # [m]
        self.Npart = Npart
        self.wallplugEfficiency = wallplugEfficiency
        self.acceleratingGradient = acceleratingGradient
        
        self.jitter = SimpleNamespace()
        self.jitter.x0 = 0
        self.jitter.y0 = 0
        self.jitter.z0 = 0
        self.jitter.t0 = 0
        
    
    def track(self, _ = None, savedepth=0, runnable=None, verbose=False):
             
        # make empty beam
        beam = Beam()
        
        # make energy spread
        sigE = self.sigE
        if self.relsigE is not None:
            if sigE is None:
                sigE = self.E0 * self.relsigE
            elif abs(self.sigE - self.E0 * self.relsigE) > 0:
                raise Exception("Both absolute and relative energy spread defined.")
           
        # Lorentz gamma
        gamma = energy2gamma(self.E0)

        # horizontal and vertical phase spaces
        xs, xps = generateTraceSpace(self.emitnx/gamma, self.betax, self.alphax, self.Npart)
        ys, yps = generateTraceSpace(self.emitny/gamma, self.betay, self.alphay, self.Npart)
        
        # longitudinal phase space
        zs = np.random.normal(loc = self.z, scale = self.sigz, size = self.Npart)
        Es = np.random.normal(loc = self.E0, scale = sigE, size = self.Npart)

        # add transverse jitters and offsets
        xs += np.random.normal(scale = self.jitter.x0) + self.x0
        ys += np.random.normal(scale = self.jitter.y0) + self.y0
        
        # add longitudinal jitters
        if self.jitter.t0 == 0:
            self.jitter.t0 = self.jitter.z0/SI.c
        if self.jitter.z0 == 0:
            self.jitter.z0 = self.jitter.t0*SI.c
        zs += np.random.normal(scale = self.jitter.z0)
        
        # create phase space
        beam.setPhaseSpace(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.Q)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def length(self):
        if self.acceleratingGradient is not None:
            return self.E0/self.acceleratingGradient
        else:
            return self.L
    
    def charge(self):
        return self.Q
    
    def energy(self):
        return self.E0
    
    def energyEfficiency(self):
        return self.wallplugEfficiency
    