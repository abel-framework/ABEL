from abc import abstractmethod
from abel import Runnable, Event, Beam
import os

class InteractionPoint(Runnable):
    
    # run simulation
    def run(self, runnable1, runnable2, run_name=None, all_by_all=False, verbose=True, overwrite=False):
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "ip_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.mkdir(self.run_path())
        else:
            if overwrite:
                self.clear_run_data()
        
        # perform tracking
        for shot1 in range(runnable1.num_shots):
            self.shot1 = shot1
            
            # get first beam
            beam1 = runnable1[shot1].final_beam()
            
            # do shot-by-shot or all-shots-by-all-shots?
            if all_by_all:
                shots2 = range(runnable2.num_shots)
            else:
                shots2 = [shot1]
            
            # go through events
            for shot2 in shots2:
                self.shot2 = shot2
                
                # load or run event
                files = self.run_data()
                if files:
                    event = Event.load(files[0], load_beams=False)
                    if verbose:
                        print(f">> EVENT {shot1+1}-{shot2+1} already exists.")
                else:
                    
                    # get second beam
                    beam2 = runnable2[shot2].final_beam()
                    
                    # get event
                    event = self.interact(beam1, beam2)
                    event.save(self)
                    
                    if verbose:
                        print(f">> EVENT {shot1+1}-{shot2+1}: Luminosity (full/peak/geom.): {event.full_luminosity()/1e34:.3} / {event.peak_luminosity()/1e34:.3} / {event.geometric_luminosity()/1e34:.2} \u03BCb^-1")
        
        # return event from last interaction
        return event
    
    
    # generate track path
    def shot_path(self, shot1=None, shot2=None):
        if shot1 is None:
            shot1 = self.shot1
        if shot2 is None:
            shot2 = self.shot2
        return self.run_path() + 'event_' + str(shot1).zfill(3) + '_' + str(shot2).zfill(3) + '/'
    
    
    # get tracking data
    #def run_data(self, shot_name=None):
    #    files = [self.run_path() + "/" + f for f in os.listdir(self.run_path()) if (os.path.isfile(os.path.join(self.run_path(), f)) and not ".DS_Store" in f)]
    #    files.sort()
    #    if shot_name is not None:
    #        files = [f for f in files if shot_name in f]
    #    return files
    
    
    # interact
    @abstractmethod
    def interact(self, beam1, beam2):
        pass