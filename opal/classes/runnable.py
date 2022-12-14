from abc import ABC
from opal import CONFIG
from os import listdir, remove, mkdir
from os.path import isfile, join, exists

class Runnable(ABC):
    
    # run simulation
    def run(self, runname=None, savedepth=2, verbose=True, overwrite=True):
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
            
        # make and clear tracking directory (if allowed)
        if not exists(self.runPath()):
            mkdir(self.runPath())
        else:
            if overwrite:
                self.clearRunData()
            else:
                raise Exception("Run folder already exists and will not be overwritten.")
            
        # perform tracking
        beam = self.track(beam=None, savedepth=savedepth, runnable=self, verbose=verbose)

        return beam
    
    
    # generate track path
    def runPath(self):
        return CONFIG.trackdata_path + self.runname + "/"
    
    
    # get tracking data
    def runData(self):
        files = [f for f in listdir(self.runPath()) if isfile(join(self.runPath(), f))]    
        files.sort()
        return files
    
    
    # clear tracking data
    def clearRunData(self):
        for file in self.runData():
            remove(self.runPath() + file)
    