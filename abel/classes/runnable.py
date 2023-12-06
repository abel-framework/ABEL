from abc import ABC
from abel import CONFIG, Beam
import os, shutil, time, sys
from datetime import datetime
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import joblib.parallel
import collections
import numpy as np
from matplotlib import pyplot as plt
import dill as pickle

class Runnable(ABC):
    
    # run simulation
    def run(self, run_name=None, num_shots=1, savedepth=2, verbose=True, overwrite=True, parallel=False, max_cores=16): 
        # TODO: implement overwrite_from=(trackable)
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.run_name = run_name
        
        # save variables
        self.num_shots = num_shots
        self.savedepth = savedepth
        self.verbose = verbose
        self.overwrite = overwrite
        
        # make base folder and clear tracking directory
        if self.overwrite or not os.path.exists(self.run_path()):
            self.clear_run_data()
        
        # perform shots (in parallel or series)
        if parallel:
            
            # recalculate number of cores used
            num_cores = min(max_cores, num_shots)
            
            # perform parallel tracking
            with joblib_progress('Tracking shots ('+str(num_cores)+' in parallel)', num_shots):
                Parallel(n_jobs=num_cores)(delayed(self.perform_shot)(shot) for shot in range(num_shots))
                time.sleep(0.1) # hack to allow printing progress
            
        else:   
            
            # perform in-series tracking
            for shot in range(num_shots):
                self.perform_shot(shot)
        
        # return final beam from first shot
        self.__dict__.update(self[0].__dict__)
        return self.final_beam()
    
    
    # shot tracking function (to be repeated)
    def perform_shot(self, shot):
        
        # set current shot
        self.shot = shot
        
        # apply scan function if it exists
        if self.is_scan():
            self.step = self.steps[shot]
            self.scan_fcn(self, self.vals_full[shot])
            #self.scan_fcn = None
        
        # check if object exists
        if not self.overwrite and os.path.exists(self.object_path(shot)):
            print('>> SHOT ' + str(shot+1) + ' already exists and will not be overwritten.', flush=True)
            
        else:

            # clear the shot folder
            self.clear_run_data(shot)

            # run tracking
            if self.num_shots > 1 and self.verbose:
                print('>> SHOT ' + str(shot+1) + '/' + str(self.num_shots), flush=True)

            #if overwrite_from is None: # TODO
            beam = self.track(beam=None, savedepth=self.savedepth, runnable=self, verbose=self.verbose)

            # save object to file
            self.save()
    
    
    # generate run folder
    def run_path(self):
        return CONFIG.run_data_path + self.run_name + '/'
    
    
    # generate object path
    def object_path(self, shot=None):
        if shot is None:
            shot = self.shot
        return self.shot_path(shot) + 'runnable' + '.obj'
    
    # save object to file
    def save(self):
        with open(self.object_path(), 'wb') as savefile:
            pickle.dump(self, savefile)
            
    # load object from file
    def load(self, shot=None):
        with open(self.object_path(shot), 'rb') as loadfile:
            obj = pickle.load(loadfile)
            return obj
    
    
    # generate track path
    def shot_path(self, shot=None):
        if shot is None:
            shot = self.shot
        if hasattr(self, 'steps'):
            step = self.steps[shot]
            shot_in_step = np.mod(shot, self.num_shots_per_step)
            return self.run_path() + 'step_' + str(step).zfill(3) + '_shot_' + str(shot_in_step).zfill(3) + '/'
        else:
            return self.run_path() + 'shot_' + str(shot).zfill(3) + '/'
    
    
    # get tracking data filenames
    def run_data(self, shot=None):
        shot_path = self.shot_path(shot)
        if os.path.exists(shot_path):
            filenames = [shot_path + f for f in os.listdir(shot_path) if (os.path.isfile(os.path.join(shot_path, f)) and not f.endswith('.obj'))]
            filenames.sort()
            return filenames
        else:
            return []
    
    
    # clear tracking data
    def clear_run_data(self, shot=None):
        
        # determine folder based on shot
        if shot is not None:
            clear_path = self.shot_path(shot)
        else:
            clear_path = self.run_path()
            
        # delete and remake folder
        if os.path.exists(clear_path):
            shutil.rmtree(clear_path)
        os.makedirs(clear_path)
        
    
    # number of beam outputs for shot
    def num_outputs(self, shot=0):
        files = self.run_data(shot)
        return len(files)
        
    
    # indexing operator (get beams out)
    def __getitem__(self, index):
        if isinstance(index, int):
            shot = index
            if shot < 0:
                shot = self.num_shots+shot
            return self.load(shot)
        elif isinstance(index, tuple) and len(index)==2:
            if self.is_scan():
                step = index[0]
                if step < 0:
                    step = self.num_steps+step
                shot_in_step = index[1]
                if shot_in_step < 0:
                    shot_in_step = self.num_shots_per_step+shot_in_step
                shot = step*self.num_shots_per_step + shot_in_step
                return self.load(shot)
            else:
                raise Exception('Not a scan')
        else:
            raise Exception('No shots')
    
    
    # load beam 
    def get_beam(self, index, shot=None):
        if shot is None:
            if hasattr(self, 'shot') and self.shot is not None:
                shot = self.shot
            else:
                shot = 0
        filenames = self.run_data(shot)
        return Beam.load(filenames[index])
        
    
    # initial beam
    def initial_beam(self, shot=None):
        return self.get_beam(0,shot)
    
    # final beam
    def final_beam(self, shot=None):
        return self.get_beam(-1,shot)
    
    
    ## SCAN FUNCTIONALITY
    
    def is_scan(self):
        return hasattr(self, 'scan_fcn')
        
    # scan function
    def scan(self, run_name=None, fcn=None, vals=None, label=None, scale=1, num_shots_per_step=1, savedepth=2, verbose=True, overwrite=True, parallel=False, max_cores=16):
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # set scan values
        self.scan_fcn = fcn
        self.vals = vals
        self.vals_full = np.repeat(vals,num_shots_per_step)
        self.steps = np.repeat(range(len(vals)),num_shots_per_step)
        self.num_steps = len(vals)
        self.num_shots_per_step = num_shots_per_step
        self.num_shots = self.num_steps*self.num_shots_per_step
        self.label = label
        self.scale = scale
        
        # perform run
        beam = self.run(run_name=self.run_name, num_shots=self.num_shots, savedepth=savedepth, verbose=verbose, overwrite=overwrite, parallel=parallel, max_cores=max_cores)
        
        return beam
    
    
    # plot value of beam parameters across a scan
    def plot_beam_function(self, beam_fcn, index=-1, label=None, scale=1, xscale='linear', yscale='linear'):
        self.plot_function(lambda obj : beam_fcn(obj.get_beam(index=index)), label=label, scale=scale, xscale=xscale, yscale=yscale)

    def plot_energy(self, index=-1):
        self.plot_beam_function(Beam.energy, scale=1e9, label='Energy (GeV)', index=index)

    def plot_charge(self, index=-1):
        self.plot_beam_function(Beam.charge, scale=1e-9, label='Charge (nC)', index=index)

    def plot_beam_size_x(self, index=-1):
        self.plot_beam_function(Beam.beam_size_x, scale=1e-3, label='Beam size, x (mm rms)', index=index)

    def plot_beam_size_y(self, index=-1):
        self.plot_beam_function(Beam.beam_size_y, scale=1e-3, label='Beam size, y (mm rms)', index=index)
        
    # plot value of beam parameters across a scan
    def plot_function(self, fcn, label=None, scale=1, xscale='linear', yscale='linear'):
        
        # extract values
        val_mean = np.empty(self.num_steps)
        val_std = np.empty(self.num_steps)
        for step in range(self.num_steps):
            
            # get values for this step
            val_output = np.empty(self.num_shots_per_step)
            for shot_in_step in range(self.num_shots_per_step):
                val_output[shot_in_step] = fcn(self[step,shot_in_step])
                
            # get step mean and error
            val_mean[step] = np.mean(val_output)
            val_std[step] = np.std(val_output)
        
        if not hasattr(self, 'scale'):
            self.scale = 1
        if not hasattr(self, 'label'):
            self.label = ''
            
        # plot evolution
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)
        
        ax.errorbar(self.vals/self.scale, val_mean/scale, val_std/scale, ls=':', capsize=5)
        ax.set_xlabel(self.label)
        ax.set_ylabel(label)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    
    def plot_waterfall(self, proj_fcn, label=None, scale=1, index=-1):

        # determine size of projection
        _, ctrs_initial = proj_fcn(self.get_beam(shot=0, index=index))
        _, ctrs_final = proj_fcn(self.get_beam(shot=-1, index=index))
        ctrs = np.linspace(min(min(ctrs_initial), min(ctrs_final)), max(max(ctrs_initial), max(ctrs_final)), len(ctrs_final))
        bins = np.append(ctrs, ctrs[-1]+(ctrs[1]-ctrs[0])) - (ctrs[1]-ctrs[0])/2
        
        # extract values
        waterfall = np.empty((self.num_shots, len(ctrs)))
        shots = range(self.num_shots)
        for shot in shots:
            proj, _ = proj_fcn(self.get_beam(shot=shot, index=index), bins=bins)
            waterfall[shot,:] = proj
        
        if not hasattr(self, 'scale'):
            self.scale = 1
        if not hasattr(self, 'label'):
            self.label = ''
          
        # plot evolution
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)

        p = ax.pcolor(shots, ctrs/scale, abs(waterfall.T/self.scale), cmap='GnBu', shading='auto')
        ax.set_xlabel('Shots')
        ax.set_ylabel(label)
        ax.set_title('Waterfall')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (a.u.)')


    def plot_waterfall_energy(self, index=-1):
        self.plot_waterfall(Beam.energy_spectrum, scale=1e9, label='Energy (GeV)', index=index)

    def plot_waterfall_current(self, index=-1):
        self.plot_waterfall(Beam.current_profile, scale=1e-15, label='Time (fs)', index=index)

    def plot_waterfall_x(self, index=-1):
        self.plot_waterfall(Beam.transverse_profile_x, scale=1e-3, label='x (mm)', index=index)

    def plot_waterfall_y(self, index=-1):
        self.plot_waterfall(Beam.transverse_profile_y, scale=1e-3, label='y (mm)', index=index)
        
    