from opal.utilities import SI
import numpy as np

class Beam():
    
    # empty beam
    def __init__(self, phasespace = None, Npart = 1000):
        
        if phasespace is not None:
            self.phasespace = phasespace
        else:
            self.phasespace = np.zeros((8, Npart))
            
        self.trackableNumber = 0
        self.stageNumber = 0
        self.location = 0        
    
    
    ## BEAM ARRAYS

    def xs(self):
        return self.phasespace[0,:]
    def xps(self):
        return self.phasespace[1,:]
    def ys(self):
        return self.phasespace[2,:]
    def yps(self):
        return self.phasespace[3,:]
    def zs(self):
        return self.phasespace[4,:]
    def Es(self):
        return self.phasespace[5,:]
    def gammas(self):
        return self.Es() * SI.e / (SI.me * SI.c**2)
    def qs(self):
        return self.phasespace[6,:]
     
        
    ## BEAM STATISTICS
    
    def charge(self):
        return np.sum(self.qs())
    
    def energy(self):
        return np.mean(self.Es())
    
    def energySpread(self):
        return np.std(self.Es())
    
    def relEnergySpread(self):
        return self.energySpread()/self.energy()
    
    def bunchLength(self):
        return np.std(self.zs())
    
    def beamSizeX(self):
        return np.std(self.xs())

    def beamSizeY(self):
        return np.std(self.ys())
    
    def normEmittanceX(self):
        return np.sqrt(np.linalg.det(np.cov(self.phasespace[0:2,:]))) * np.mean(self.gammas())
    
    def normEmittanceY(self):
        return np.sqrt(np.linalg.det(np.cov(self.phasespace[2:4,:]))) * np.mean(self.gammas())
        
        
    ## CHANGE BEAM
    
    def accelerate(self, deltaE, chirp = 0):
        self.phasespace[5,:] += deltaE + self.zs() * chirp
        
    def compress(self, R56, E0):
        self.phasespace[4,:] += (1-self.Es()/E0) * R56
        
    def adiabaticDamping(self, E0, E):
        self.phasespace[0:4,:] *= np.sqrt(E0/E)
        
    def flipTransversePhaseSpaces(self):
        self.phasespace[1,:] *= -1
        self.phasespace[3,:] *= -1
    
  
    ## SAVE AND LOAD BEAM
    
    # make filename for beam 
    def filename(self):
        return "beam_" + str(self.trackableNumber).zfill(3) + "_"  + str(self.stageNumber).zfill(3) + "_" + "{:012.6F}".format(self.location) + ".txt"
      
    # save beam
    def save(self, linac):
        np.savetxt(linac.trackingfolder + self.filename(), self.phasespace)
    
    # load beam
    @classmethod
    def load(_, filename):
        
        # create beam from phase space
        beam = Beam(phasespace = np.loadtxt(filename))
            
        # find stage and location from filename
        parts = filename.split('_')
        beam.trackableNumber = int(parts[1])
        beam.stageNumber = int(parts[2])
        subparts = parts[3].split('.')
        beam.location = float(subparts[0] + '.' + subparts[1])
        
        return beam
      