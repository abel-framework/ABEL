from abc import ABC, abstractmethod

class Trackable(ABC):
    
    # tracking function
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # remove nan particles
        beam.remove_nans()
        
        # advance beam location
        beam.location += self.get_length()
        
        # iterate trackable number
        beam.trackable_number += 1
        
        # save beam to file
        if savedepth >= 0:
            
            # describe tracking
            if verbose:
                print(f"Tracking element {beam.trackable_number+1} (s = {beam.location:.1f} m, {beam.charge()*1e9:.2f} nC, {beam.energy()/1e9:.1f} GeV, {type(self).__name__}, stage {beam.stage_number})")
                
            # save to file
            if runnable is not None:
                beam.save(runnable)
            
        return beam
    
    
    # length of the trackable element
    @abstractmethod
    def get_length(self):
        pass
    
    # abbreviation of the get_length() function
    def __len__(self):
        return self.get_length()
    
    
    # object for survey plotting
    @abstractmethod
    def survey_object(self):
        pass
        
    