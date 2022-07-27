from opal import Beam, Source
from opal.utilities import SI
from opal.utilities.beamphysics import generateTraceSpace
import numpy as np

class SourceBasic(Source):
    
    def __init__(self, E = None, Q = None, sigE = None, sigz = None, emitnx = None, emitny = None, betax = None, betay = None, alphax = 0, alphay = 0, L = 0, Npart = 1000):
        self.E = E
        self.Q = Q
        self.sigE = sigE # [eV]
        self.sigz = sigz # [m]
        self.Npart = Npart
        self.emitnx = emitnx # [m rad]
        self.emitny = emitny # [m rad]
        self.betax = betax # [m]
        self.betay = betay # [m]
        self.alphax = alphax # [m]
        self.alphay = alphay # [m]
        self.L = L # [m]
        self.Npart = Npart
        
    def track(self, _):
        
        # create empty beam
        beam = Beam(Npart = self.Npart)
            
        # Lorentz gamma
        gamma = self.energy() * SI.e / (SI.me * SI.c**2)

        # horizontal phase space
        xs, xps = generateTraceSpace(self.emitnx/gamma, self.betax, self.alphax, self.Npart)
        beam.phasespace[0,:] = xs
        beam.phasespace[1,:] = xps

        # vertical phase space
        ys, yps = generateTraceSpace(self.emitny/gamma, self.betay, self.alphay, self.Npart)
        beam.phasespace[2,:] = ys
        beam.phasespace[3,:] = yps

        # longitudinal phase space
        beam.phasespace[4,:] = np.random.normal(scale = self.sigz, size = self.Npart)
        beam.phasespace[5,:] = np.random.normal(loc = self.E, scale = self.sigE, size = self.Npart)

        # charge
        beam.phasespace[6,:] = self.Q/self.Npart

        return super().track(beam)
    
    def length(self):
        return self.L
    
    def energy(self):
        return self.E