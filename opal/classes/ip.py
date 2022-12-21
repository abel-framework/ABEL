from abc import abstractmethod
from opal import Runnable, Event, Beam
from os import mkdir
from os.path import exists

class InteractionPoint(Runnable):
    
    # run simulation
    def run(self, runnable1, runnable2, runname=None, allByAll=False, verbose=True, overwrite=True):
        
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

                # make and clear tracking directory
                if not exists(self.shotPath()):
                    mkdir(self.shotPath())
                else:
                    if overwrite:
                        self.clearRunData(n)
                    else:
                        print("Event #" + str(n) + " already exists and will not be overwritten.")
                        #files = self.runData(self.shotname)
                        files = '' # TODO
                        event = Event.load(files[0][0])
                        continue
                
                # run tracking
                if len(runnable1.shotnames) > 1 and verbose:
                    print(">> EVENT #" + str(n))
                
                # get second beam
                beam2 = Beam.load(runnable2.runData(runnable2.shotnames[j])[-1][-1])
                
                # get event
                event = self.interact(beam1, beam2)
                
                # increment event number
                n += 1
        
        # return event from last interaction
        return event
    
    
    # interact
    @abstractmethod
    def interact(self, beam1, beam2):
        pass