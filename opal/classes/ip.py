from abc import abstractmethod
from opal import Runnable, Event, Beam
from os import listdir, mkdir
from os.path import isfile, join, exists

class InteractionPoint(Runnable):
    
    # run simulation
    def run(self, runnable1, runnable2, runname=None, allByAll=False, verbose=True, overwrite=False):
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "ip_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
        
        # declare shots list
        self.shotnames = []
        
        # make base folder and clear tracking directory
        if not exists(self.runPath()):
            mkdir(self.runPath())
        
        # make base folder and clear tracking directory
        if not exists(self.runPath()):
            mkdir(self.runPath())
        else:
            if overwrite:
                self.clearRunData()
        
        # perform tracking
        existing_events = []
        existing_flag = False
        n = 1
        for i in range(len(runnable1.shotnames)):
            
            # get first beam
            beam1 = Beam.load(runnable1.runData(runnable1.shotnames[i])[-1][-1])
            
            # do shot-by-shot or all-shots-by-all-shots?
            if allByAll:
                js = range(len(runnable2.shotnames))
            else:
                js = [i]
            
            # go through events
            for j in js:
                
                # make shot folder
                self.shotname = "/event_" + str(n) + "_shots_" + str(i+1) + "_" + str(j+1)

                # add to shots list
                self.shotnames.append(self.shotname)

                # load or run event
                files = self.runData(self.shotname)
                if files:
                    existing_events.append(n)
                    event = Event.load(files[0], loadBeams=False)
                else:
                    
                    # flush message of existing events
                    if verbose and len(existing_events) > 0:
                        if len(existing_events) == 1:
                            print("Event #" + str(n) + " already exists.")
                        else:
                            print("Events #" + str(min(existing_events)) + "-" + str(max(existing_events)) + " already exist.")
                        existing_events = []    
                        existing_flag = True
                        
                    # run tracking
                    #if len(runnable1.shotnames) > 1 and verbose:
                        #print(">> EVENT #" + str(n))
                    
                    # get second beam
                    beam2 = Beam.load(runnable2.runData(runnable2.shotnames[j])[-1][-1])
                    
                    # get event
                    event = self.interact(beam1, beam2)
                    event.save(self)
                    
                    if verbose:
                        #print(f"Luminosity: {event.fullLuminosity()/1e34:.3}/\u03BCb (full), {event.peakLuminosity()/1e34:.3}/\u03BCb (peak), {event.geometricLuminosity()/1e34:.3}/\u03BCb (geom.)")
                        print(f">> EVENT #{n}: Luminosity (full/peak/geom.): {event.fullLuminosity()/1e34:.3} / {event.peakLuminosity()/1e34:.3} / {event.geometricLuminosity()/1e34:.2} \u03BCb^-1")
                
                # increment event number
                n += 1
        
        # print
        if existing_events and not existing_flag:
            print("All events (#" + str(min(existing_events)) + "-" + str(max(existing_events)) + ") already exist.")
        
        # return event from last interaction
        return event
    
    # get tracking data
    def runData(self, shotname=None):
        files = [self.runPath() + "/" + f for f in listdir(self.runPath()) if (isfile(join(self.runPath(), f)) and not ".DS_Store" in f)]
        files.sort()
        if shotname is not None:
            files = [f for f in files if shotname in f]
        return files
    
    # interact
    @abstractmethod
    def interact(self, beam1, beam2):
        pass