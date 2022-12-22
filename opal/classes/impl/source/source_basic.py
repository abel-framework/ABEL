from opal import Source, Beam
from opal.utilities import SI
from opal.utilities.beamphysics import generateTraceSpace
from opal.utilities.relativity import energy2gamma
import numpy as np

class SourceBasic(Source):
    
    def __init__(self, E0 = None, Q = None, sigE = None, sigz = None, z = 0, emitnx = None, emitny = None, betax = None, betay = None, alphax = 0, alphay = 0, L = 0, Npart = 1000):
        self.E0 = E0
        self.Q = Q
        self.sigE = sigE # [eV]
        self.sigz = sigz # [m]
        self.z = z # [m]
        self.Npart = Npart
        self.emitnx = emitnx # [m rad]
        self.emitny = emitny # [m rad]
        self.betax = betax # [m]
        self.betay = betay # [m]
        self.alphax = alphax # [m]
        self.alphay = alphay # [m]
        self.L = L # [m]
        self.Npart = Npart
    
    
    def track(self, _ = None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.E0)

        # horizontal and vertical phase spaces
        xs, xps = generateTraceSpace(self.emitnx/gamma, self.betax, self.alphax, self.Npart)
        ys, yps = generateTraceSpace(self.emitny/gamma, self.betay, self.alphay, self.Npart)
        
        # longitudinal phase space
        zs = np.random.normal(loc = self.z, scale = self.sigz, size = self.Npart)
        Es = np.random.normal(loc = self.E0, scale = self.sigE, size = self.Npart)

        # create phase space
        beam.setPhaseSpace(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.Q)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def length(self):
        return self.L
    
    
    def energy(self):
        return self.E0