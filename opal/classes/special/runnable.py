from abc import ABC
from opal import CONFIG, Beam
import os
from datetime import datetime

class Runnable(ABC):
    
    # run simulation
    def run(self, runname=None, shots=1, savedepth=2, verbose=True, overwrite=True):
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
        
        # declare shots list
        self.shotnames = []
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.runPath()):
            os.makedirs(self.runPath(), exist_ok=True)
        else:
            if overwrite:
                self.clearRunData()
        
        # perform tracking
        for i in range(1, shots+1):
            
            # make shot folder
            self.shotname = "/shot_" + str(i)
            
            # add to shots list
            self.shotnames.append(self.shotname)
            
            # make and clear tracking directory
            if not os.path.exists(self.shotPath()):
                os.mkdir(self.shotPath())
            else:
                if overwrite:
                    self.clearRunData(i)
                else:
                    print("Shot #" + str(i) + " already exists and will not be overwritten.")
                    files = self.runData(self.shotname)
                    beam = Beam.load(files[0][-1])
                    continue

            # run tracking
            if shots > 1 and verbose:
                print(">> SHOT #" + str(i))  
            beam = self.track(beam=None, savedepth=savedepth, runnable=self, verbose=verbose)
                

        # return beam from last shot
        return beam
    
    
    # generate run path(s)
    def runPath(self):
        return CONFIG.rundata_path + self.runname
    
    # generate track path(s)
    def shotPath(self):
        return self.runPath() + self.shotname
    
    # generate track path(s)
    def shotPathsAll(self):
        paths = []
        for shotname in self.shotnames:
            paths.append(self.runPath() + shotname)
        return paths
    
    
    # get tracking data
    def runData(self, shotname=None):
        
        # collect shots
        if shotname is None:
            shotPaths = self.shotPathsAll()
        else:
            shotPaths = [self.shotPathsAll()[self.shotnames.index(shotname)]]
        
        # find filenames
        files = []
        for i in range(len(shotPaths)):
            shotFiles = [shotPaths[i] + "/" + f for f in os.listdir(shotPaths[i]) if os.path.isfile(os.path.join(shotPaths[i], f))]
            shotFiles.sort()
            files.append(shotFiles)
        
        return files
    
    
    # clear tracking data
    def clearRunData(self, shotname=None):
        if shotname is not None:
            files = self.runData(shotname)
            for shotFiles in files:
                for file in shotFiles:
                    os.remove(file)
        else:
            for folder in os.listdir(self.runPath()):
                path = self.runPath() + "/" + folder
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        os.remove(path + "/" + file)
                    os.rmdir(path)
                else:
                    os.remove(path)
    