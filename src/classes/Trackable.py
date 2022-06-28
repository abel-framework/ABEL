from abc import ABC, abstractmethod

class Trackable(ABC):
    
    def track(self, beam):
        beam.trackableNumber += 1
        beam.location += self.length()
        return beam
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def plotObject(self):
        pass
        
    