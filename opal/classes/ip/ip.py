from abc import abstractmethod
from opal import Runnable, Event, Beam
import os

class InteractionPoint(Runnable):
    
    # run simulation
    def run(self, runnable1, runnable2, run_name=None, all_by_all=False, verbose=True, overwrite=False):
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "ip_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # declare shots list
        self.shot_names = []
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.mkdir(self.run_path())
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.mkdir(self.run_path())
        else:
            if overwrite:
                self.clear_run_data()
        
        # perform tracking
        existing_events = []
        existing_flag = False
        n = 1
        for i in range(len(runnable1.shot_names)):
            
            # get first beam
            beam1 = Beam.load(runnable1.run_data(runnable1.shot_names[i])[-1][-1])
            
            # do shot-by-shot or all-shots-by-all-shots?
            if all_by_all:
                js = range(len(runnable2.shot_names))
            else:
                js = [i]
            
            # go through events
            for j in js:
                
                # make shot folder
                self.shot_name = "/event_" + str(n) + "_shots_" + str(i+1) + "_" + str(j+1)

                # add to shots list
                self.shot_names.append(self.shot_name)

                # load or run event
                files = self.run_data(self.shot_name)
                if files:
                    existing_events.append(n)
                    event = Event.load(files[0], load_beams=False)
                else:
                    
                    # flush message of existing events
                    if verbose and len(existing_events) > 0:
                        if len(existing_events) == 1:
                            print("Event #" + str(n) + " already exists.")
                        else:
                            print("Events #" + str(min(existing_events)) + "-" + str(max(existing_events)) + " already exist.")
                        existing_events = []    
                        existing_flag = True
                    
                    # get second beam
                    beam2 = Beam.load(runnable2.run_data(runnable2.shot_names[j])[-1][-1])
                    
                    # get event
                    event = self.interact(beam1, beam2)
                    event.save(self)
                    
                    if verbose:
                        print(f">> EVENT #{n}: Luminosity (full/peak/geom.): {event.full_luminosity()/1e34:.3} / {event.peak_luminosity()/1e34:.3} / {event.geometric_luminosity()/1e34:.2} \u03BCb^-1")
                
                # increment event number
                n += 1
        
        # print
        if existing_events and not existing_flag:
            print("All events (#" + str(min(existing_events)) + "-" + str(max(existing_events)) + ") already exist.")
        
        # return event from last interaction
        return event
    
    # get tracking data
    def run_data(self, shot_name=None):
        files = [self.run_path() + "/" + f for f in os.listdir(self.run_path()) if (os.path.isfile(os.path.join(self.run_path(), f)) and not ".DS_Store" in f)]
        files.sort()
        if shot_name is not None:
            files = [f for f in files if shot_name in f]
        return files
    
    # interact
    @abstractmethod
    def interact(self, beam1, beam2):
        pass