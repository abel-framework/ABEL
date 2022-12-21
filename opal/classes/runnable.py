from abc import ABC
from opal import CONFIG, Beam
from os import listdir, remove, mkdir, rmdir
from os.path import isfile, isdir, join, exists

class Runnable(ABC):
    
    # run simulation
    def run(self, runname=None, shots=1, savedepth=2, verbose=True, overwrite=True):
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
        
        # declare shots list
        self.shots = []
        
        # make base folder and clear tracking directory
        if not exists(self.runPath()):
            mkdir(self.runPath())
        else:
            if overwrite:
                self.clearRunData()
            else:
                print("Run folder already exists and will not be overwritten.")
        
        # perform tracking
        for i in range(shots):
            
            # make shot folder
            self.shotname = "/shot_" + str(i)
            
            # add to shots list
            self.shots.append(self.shotname)
            
            # make and clear tracking directory
            if not exists(self.shotPath()):
                mkdir(self.shotPath())
            else:
                if overwrite:
                    self.clearRunData(i)
                else:
                    print("Shot #" + str(i) + " already exists and will not be overwritten.")
                    files = self.runData(i)
                    beam = Beam.load(files[0][-1])
                    continue

            # run tracking
            if shots > 1:
                print(">> SHOT #" + str(i))
                
            beam = self.track(beam=None, savedepth=savedepth, runnable=self, verbose=verbose)
                

        # return beam from last run
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
        for i in range(len(self.shots)):
            paths.append(self.runPath() + self.shots[i])
        return paths
    
    
    # get tracking data
    def runData(self, shot=None):
        
        # collect shots
        if shot is None:
            shotPaths = self.shotPathsAll()
        else:
            shotPaths = [self.shotPathsAll()[shot]]
        
        # find filenames
        files = []
        for i in range(len(shotPaths)):
            shotFiles = [shotPaths[i] + "/" + f for f in listdir(shotPaths[i]) if isfile(join(shotPaths[i], f))]
            shotFiles.sort()
            files.append(shotFiles)
        
        return files
    
    
    # clear tracking data
    def clearRunData(self, shot=None):
        
        # get files
        if shot is not None:
            files = self.runData(shot)
            for shotFiles in files:
                for file in shotFiles:
                    remove(file)
        else:
            for folder in listdir(self.runPath()):
                path = self.runPath() + "/" + folder
                if isdir(path):
                    for file in listdir(path):
                        remove(path + "/" + file)
                    rmdir(path)
                else:
                    remove(path)
            
        # delete files
    