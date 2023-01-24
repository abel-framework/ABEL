from abc import ABC, abstractmethod

class Trackable(ABC):
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # advance beam location
        beam.location += self.length()
        
        # iterate trackable number
        beam.trackableNumber += 1
        
        # save beam to file
        if savedepth >= 0:
            
            # describe tracking
            if verbose:
                print(f"Tracking element #{beam.trackableNumber} (s = {beam.location:.1f} m, {beam.charge()*1e9:.1f} nC, {beam.energy()/1e9:.1f} GeV, {type(self).__name__}, stage {beam.stageNumber})")
                
            # save to file
            if runnable is not None:
                beam.save(runnable)
            
        return beam
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def plotObject(self):
        pass
        
    