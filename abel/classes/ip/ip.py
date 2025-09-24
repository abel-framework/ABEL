from abc import abstractmethod
from abel.classes.runnable import Runnable
from abel.classes.event import Event
from abel.classes.beam import Beam
from abel.classes.cost_modeled import CostModeled
import os, shutil
import numpy as np

class InteractionPoint(Runnable, CostModeled):
    
    def __init__(self, num_ips=1, gamma_gamma=False):
        self.num_ips = num_ips
        self.gamma_gamma = gamma_gamma
        super().__init__()
    
    # run simulation
    def run(self, runnable1, runnable2, run_name=None, all_by_all=False, verbose=True, overwrite=False, step_filter=None):
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "ip_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.mkdir(self.run_path())

        shots_to_perform = np.arange(runnable1.num_shots)
        if step_filter is not None:
            shots_to_perform = shots_to_perform[np.isin(runnable1.steps, step_filter)]
        
        # perform tracking
        for shot1 in shots_to_perform:
            self.shot1 = int(shot1)
            
            # get first beam
            try:
                beam1 = runnable1.load(self.shot1).final_beam
            except:
                continue
            
            # do shot-by-shot or all-shots-by-all-shots?
            if all_by_all:
                shots2 = np.arange(runnable2.num_shots)
                step_filter2 = runnable1[self.shot1].step
                shots2 = shots2[np.isin(runnable2.steps, step_filter2)]
            else:
                shots2 = [self.shot1]
            
            # go through events
            for shot2 in shots2:
                self.shot2 = int(shot2)
                
                if overwrite:
                    self.clear_run_data(shot1, shot2)
                
                # load or run event
                files = self.run_data()
                if self.shot_path()+'event.h5' in files:
                    event = Event.load(files[0], load_beams=False)
                    if verbose:
                        print(f">> EVENT {self.shot1+1}-{self.shot2+1} already exists.")
                else:
                    
                    # get second beam
                    try:
                        beam2 = runnable2[self.shot2].final_beam
                    except:
                        continue
                           
                    # the step has to be the same
                    if runnable1[self.shot1].step != runnable2[self.shot2].step:
                        continue
                    
                    # get event
                    event = self.interact(beam1, beam2)
                    event.save(self, shot1=shot1, shot2=shot2)
                    
                    if verbose:
                        if not self.gamma_gamma:
                            print(f">> EVENT {self.shot1+1}-{self.shot2+1}: Luminosity (full/peak/geom.): {event.full_luminosity()/1e34:.3} / {event.peak_luminosity()/1e34:.3} / {event.geometric_luminosity()/1e34:.2} \u03BCb^-1")
                        else:
                            print(f">> EVENT (gamma-gamma) {self.shot1+1}-{self.shot2+1}: Luminosity (full/peak/geom.): {event.full_luminosity()/1e34:.3} / {event.peak_luminosity()/1e34:.3} / {event.geometric_luminosity()/1e34:.2} \u03BCb^-1")
                    
        # return event from last interaction
        return event
    
    # clear tracking data
    def clear_run_data(self, shot1=None, shot2=None):
        
        # determine folder based on shot
        if shot1 is not None and shot2 is not None:
            clear_path = self.shot_path(shot1, shot2)
        else:
            clear_path = self.run_path()
            
        # delete and remake folder
        if os.path.exists(clear_path):
            shutil.rmtree(clear_path)
        os.makedirs(clear_path)
        
    # generate track path
    def shot_path(self, shot1=None, shot2=None):
        if shot1 is None:
            shot1 = self.shot1
        if shot2 is None:
            shot2 = self.shot2
        return self.run_path() + 'event_' + str(shot1).zfill(3) + '_' + str(shot2).zfill(3) + '/'
    
    
    # get tracking data
    def run_data(self, shot_name=None):
        files = [self.run_path() + f + "/event.h5" for f in os.listdir(self.run_path())]
        files.sort()
        if shot_name is not None:
            files = [f for f in files if shot_name in f]
        return files
    
    
    def get_cost_breakdown(self):
        return (f'Interaction point ({self.num_ips}x)', CostModeled.cost_per_ip * self.num_ips)
    
    # interact
    @abstractmethod
    def interact(self, beam1, beam2):
        pass