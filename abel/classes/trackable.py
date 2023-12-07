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
                print(f"Tracking element {beam.trackable_number+1} ({type(self).__name__}, stage {beam.stage_number}, s = {beam.location:.1f} m, {beam.energy()/1e9:.1f} GeV, {beam.charge()*1e9:.2f} nC, {beam.rel_energy_spread()/1e-2:.1f}% rms, {beam.norm_emittance_x()/1e-6:.1f}/{beam.norm_emittance_y()/1e-6:.1f} µm-rad)")
                
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
        
    