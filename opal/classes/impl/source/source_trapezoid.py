from opal import Source, Beam
from opal.utilities import SI
from opal.utilities.beamphysics import generateTraceSpace
from opal.utilities.relativity import energy2gamma
import numpy as np

class SourceTrapezoid(Source):
    
    def __init__(self, E0 = None, Q = None, sigE = None, deltaz = None, Ihead = 0, z = 0, emitnx = None, emitny = None, betax = None, betay = None, alphax = 0, alphay = 0, L = 0, Npart = 1000):
        self.E0 = E0
        self.Q = Q
        self.sigE = sigE # [eV]
        self.deltaz = deltaz # [m]
        self.z = z # [m]
        self.Ihead = Ihead # [A]
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
        
        # longitudinal positions
        Q_uniform = abs(self.Ihead) * self.deltaz / SI.c
        if Q_uniform > 2*abs(self.Q):
            Q_triangle = abs(self.Q)
            Q_uniform = 0
            zmode = self.z
        elif abs(self.Q) > Q_uniform:
            Q_triangle = abs(self.Q) - Q_uniform
            zmode = self.z - self.deltaz
        else:
            Q_triangle = Q_uniform - abs(self.Q)
            Q_uniform = abs(self.Q) - Q_triangle
            zmode = self.z
            
        index_split = round(self.Npart*abs(Q_uniform)/abs(self.Q))
        inds = np.random.permutation(self.Npart)
        mask_uniform = inds[0:index_split]
        mask_triangle = inds[index_split:self.Npart]
        zs = np.zeros(self.Npart)
        zs[mask_uniform] = np.random.uniform(low = self.z - self.deltaz, high = self.z, size = len(mask_uniform))
        zs[mask_triangle] = np.random.triangular(left = self.z - self.deltaz, right = self.z, mode = zmode, size = len(mask_triangle))
        
        # energies
        Es = np.random.normal(loc = self.E0, scale = self.sigE, size = self.Npart)

        # create phase space
        beam.setPhaseSpace(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.Q)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def length(self):
        return self.L
    
    
    def energy(self):
        return self.E0