from opal import CONFIG, Runnable, Beam
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

class Scannable(Runnable):
    
    # run scan simulation
    def scan(self, run_name=None, fcn=None, vals=None, num_shots_per_step=1, savedepth=2, verbose=True, overwrite=True):
        
        # save variables
        self.fcn = fcn
        self.vals = vals
        self.num_steps = len(vals)
        self.num_shots_per_step = num_shots_per_step
        self.num_shots = self.num_steps*self.num_shots_per_step
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
        for step in range(len(vals)):
            
            if verbose:
                print('>> STEP ' + str(step+1) + '/' + str(len(vals)))
            
            # call step function
            output = fcn(self, vals[step])
            self.label = None
            self.scale = 1
            if isinstance(output,tuple):
                if len(output) == 2:
                    self, self.label = output
                if len(output) == 3:
                    self, self.label, self.scale = output
            else:
                self
            
            for shot_in_step in range(num_shots_per_step):
                
                # make shot folder
                self.shot_name = '/step_' + str(step) + '_shot_' + str(shot_in_step)

                # add to shots list
                self.shot_names.append(self.shot_name)

                # make and clear tracking directory
                if not os.path.exists(self.shot_path()):
                    os.mkdir(self.shot_path())
                else:
                    if overwrite:
                        self.clear_run_data(shot_in_step+1)
                    else:
                        print('>>>> SHOT ' + str(shot_in_step+1) + ' already exists and will not be overwritten.')
                        files = self.run_data(self.shot_name)
                        beam = Beam.load(files[0][-1])
                        continue
                
                # run tracking
                if num_shots_per_step > 1 and verbose:
                    shot = num_shots_per_step*step + shot_in_step
                    print('>>>> SHOT ' + str(shot_in_step+1) + '/' + str(num_shots_per_step) + ' (' + str(shot+1) + '/' + str(num_shots_per_step*len(vals)) +')')
                beam = self.track(beam=None, savedepth=savedepth, runnable=self, verbose=verbose)


        # return beam from last shot
        return beam
    
    
    # indexing operator (get beams out)
    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index)==3 and hasattr(self, 'num_shots_per_step'):
            beam_index = index[0]
            shot = index[2]*self.num_shots_per_step + index[1]
            files = self.run_data()
            return Beam.load(files[shot][beam_index])
        else:
            return super().__getitem__(index)
    
    
    # initial beam
    def initial_beam(self, shot=0, step=0):
        return self[0,shot,step]
    
    # final beam
    def final_beam(self, shot=0, step=0):
        return self[-1,shot,step]
    
    # plot value of beam parameters across a scan
    def plot_scan(self, beam_fcn, label=None, scale=1, xscale='linear', yscale='linear'):
        
        # extract values
        val_mean = np.empty(self.num_steps)
        val_std = np.empty(self.num_steps)
        for step in range(self.num_steps):
            
            # get values for this step
            val_output = np.empty(self.num_shots_per_step)
            for shot_in_step in range(self.num_shots_per_step):
                val_output[shot_in_step] = beam_fcn(self.final_beam(shot=shot_in_step, step=step))
                
            # get step mean and error
            val_mean[step] = np.mean(val_output)
            val_std[step] = np.std(val_output)
        
        # plot evolution
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)
        
        ax.errorbar(self.vals/self.scale, val_mean/scale, val_std/scale, ls=':', capsize=5)
        ax.set_xlabel(self.label)
        ax.set_ylabel(label)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        
        
        