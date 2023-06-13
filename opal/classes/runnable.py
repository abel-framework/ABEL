from abc import ABC
from opal import CONFIG, Beam
import os
from datetime import datetime

class Runnable(ABC):
    
    # run simulation
    def run(self, run_name=None, shots=1, savedepth=2, verbose=True, overwrite=True):
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # declare shots list
        self.shot_names = []
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.makedirs(self.run_path(), exist_ok=True)
        else:
            if overwrite:
                self.clear_run_data()
        
        # perform tracking
        for i in range(1, shots+1):
            
            # make shot folder
            self.shot_name = "/shot_" + str(i)
            
            # add to shots list
            self.shot_names.append(self.shot_name)
            
            # make and clear tracking directory
            if not os.path.exists(self.shot_path()):
                os.mkdir(self.shot_path())
            else:
                if overwrite:
                    self.clear_run_data(i)
                else:
                    print("Shot #" + str(i) + " already exists and will not be overwritten.")
                    files = self.run_data(self.shot_name)
                    beam = Beam.load(files[0][-1])
                    continue

            # run tracking
            if shots > 1 and verbose:
                print(">> SHOT #" + str(i))  
            beam = self.track(beam=None, savedepth=savedepth, runnable=self, verbose=verbose)
                

        # return beam from last shot
        return beam
    
    
    # generate run path(s)
    def run_path(self):
        return CONFIG.run_data_path + self.run_name
    
    # generate track path(s)
    def shot_path(self):
        return self.run_path() + self.shot_name
    
    # generate track path(s)
    def shot_paths_all(self):
        paths = []
        for shot_name in self.shot_names:
            paths.append(self.run_path() + shot_name)
        return paths
    
    
    # get tracking data
    def run_data(self, shot_name=None):
        
        # collect shots
        if shot_name is None:
            shot_paths = self.shot_paths_all()
        else:
            shot_paths = [self.shot_paths_all()[self.shot_names.index(shot_name)]]
        
        # find filenames
        files = []
        for i in range(len(shot_paths)):
            shot_files = [shot_paths[i] + "/" + f for f in os.listdir(shot_paths[i]) if os.path.isfile(os.path.join(shot_paths[i], f))]
            shot_files.sort()
            files.append(shot_files)
        
        return files
    
    
    # clear tracking data
    def clear_run_data(self, shot_name=None):
        if shot_name is not None:
            files = self.run_data(shot_name)
            for shot_files in files:
                for file in shot_files:
                    os.remove(file)
        else:
            for folder in os.listdir(self.run_path()):
                path = self.run_path() + "/" + folder
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        os.remove(path + "/" + file)
                    os.rmdir(path)
                else:
                    os.remove(path)
    