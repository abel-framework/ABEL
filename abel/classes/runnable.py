from abc import ABC, abstractmethod
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam
import os, shutil, time, sys, csv
from datetime import datetime
import numpy as np

class Runnable(ABC):
    
    # shot tracking function (to be repeated)
    def perform_shot(self, shot):
        
        self.shot = shot
        self.step = self.steps[shot]
        
        # apply scan function if it exists
        if self.scan_fcn is not None:
            vals_all = np.repeat(self.vals,self.num_shots_per_step)
            self.scan_fcn(self, vals_all[shot])
        
        # check if object exists
        if not self.overwrite and os.path.exists(self.object_path(shot)):
            print('>> SHOT ' + str(shot+1) + ' already exists and will not be overwritten.', flush=True)
            
        else:

            # clear the shot folder
            self.clear_run_data(shot)

            # run tracking
            if self.num_shots > 1 and self.verbose:
                print('>> SHOT ' + str(shot+1) + '/' + str(self.num_shots), flush=True)

            # if overwrite_from is None: # TODO
            self.track(beam=None, savedepth=self.savedepth, runnable=self, verbose=self.verbose)

            # save object to file
            self.save()

    ## SCAN FUNCTIONALITY
    
    def is_scan(self):
        return self.scan_fcn is not None
      
    # scan function
    def scan(self, run_name=None, fcn=None, vals=[None], label=None, scale=1, num_shots_per_step=1, step_filter=None, shot_filter=None, savedepth=2, verbose=None, overwrite=False, parallel=False, max_cores=16):

        from joblib import Parallel, delayed
        from joblib_progress import joblib_progress
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        self.overwrite = overwrite
        self.verbose = verbose
        self.savedepth = savedepth
        
        if self.overwrite:
            self.clear_run_data()
            self.overwrite = False
            
        # default verbosity
        if self.verbose is None:
            self.verbose = not parallel
        
        # set scan values
        self.scan_fcn = fcn
        self.vals = vals
        self.steps = np.repeat(range(len(vals)), num_shots_per_step)
        self.num_steps = len(vals)
        self.num_shots_per_step = num_shots_per_step
        self.num_shots = self.num_steps*self.num_shots_per_step
        self.label = label
        self.scale = scale
        
        # make temp folder and run path if not existing
        if not os.path.exists(CONFIG.temp_path):
            os.makedirs(CONFIG.temp_path, exist_ok=True)
        if not os.path.exists(self.run_path()):
            os.makedirs(self.run_path(), exist_ok=True)

        # make base folder and clear tracking directory
        if step_filter == 0:
            if self.overwrite or not os.path.exists(self.run_path()):
                self.clear_run_data()
            
        # define what shots to perform
        shots_to_perform = np.arange(self.num_shots)
        if step_filter is not None:
            shots_to_perform = shots_to_perform[np.isin(self.steps, step_filter)]
        elif shot_filter is not None:
            shots_to_perform = shots_to_perform[np.isin(shots_to_perform, shot_filter)]

        # perform shots (in parallel or series)
        if parallel:
            
            # recalculate number of cores used
            num_cores = min(max_cores, len(shots_to_perform))
            
            # perform parallel tracking
            with joblib_progress('Tracking shots ('+str(num_cores)+' in parallel)', len(shots_to_perform)):
                Parallel(n_jobs=num_cores)(delayed(self.perform_shot)(shot) for shot in shots_to_perform)
                time.sleep(0.05) # hack to allow printing progress
            
        else:   
            
            # perform in-series tracking
            for shot in shots_to_perform:
                self.perform_shot(shot)
        
        # return final beam from first shot
        self.__dict__.update(self.load(shot=shots_to_perform[0]).__dict__)

    
    # run simulation
    def run(self, run_name=None, num_shots=1, savedepth=2, verbose=None, overwrite=False, parallel=False, max_cores=16): 
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.run_name = run_name
        
        # perform a scan with only one step
        self.scan(run_name=self.run_name, num_shots_per_step=num_shots, savedepth=savedepth, verbose=verbose, overwrite=overwrite, parallel=parallel, max_cores=max_cores)
    
    

    
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
        import dill as pickle
        with open(self.object_path(), 'wb') as savefile:
            pickle.dump(self, savefile)
            
    # load object from file
    def load(self, shot=None):
        import dill as pickle
        with open(self.object_path(shot), 'rb') as loadfile:
            try:
                obj = pickle.load(loadfile)
                return obj
            except:
                return None
    
    
    # generate track path
    def shot_path(self, shot=None):
        if shot is None:
            shot = self.shot
        if self.is_scan(): # if a scan
            step = self.steps[shot]
            shot_in_step = np.mod(shot, self.num_shots_per_step)
            return self.run_path() + 'step_' + str(step).zfill(3) + '_shot_' + str(shot_in_step).zfill(3) + '/'
        else:
            return self.run_path() + 'shot_' + str(shot).zfill(3) + '/'
    
    
    # get tracking data filenames
    def run_data(self, shot=None):
        shot_path = self.shot_path(shot)
        if os.path.exists(shot_path):
            filenames = [shot_path + f for f in os.listdir(shot_path) if (os.path.isfile(os.path.join(shot_path, f)) and f.startswith('beam_') and not f.endswith('.obj'))]
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
                shot = int(self.shot)
            else:
                shot = 0
        filenames = self.run_data(shot)
        return Beam.load(filenames[index])
        
    
    # initial beam
    @property
    def initial_beam(self):
        return self.get_beam(0)
    
    # final beam
    @property
    def final_beam(self):
        return self.get_beam(-1)

    
    ## Support for Bayesian optimisation through Ax.optimize
    
    def set_parameters(self, params):
        for key in params.keys():
            self.set_attr(key, params[key])
            
    # Setting attribute
    def set_attr(self, attr, val):
        pre, _, post = attr.rpartition('.') # Splits attr into post containing the last nested atrtibute and pre containing all previous attributes.
        return setattr(self.get_nested_attr(pre) if pre else self, post, val)  # Sets val to the post attribute if attr is not nested. If nested, call get_nested_attr(pre) to set val correctly.

    # Get nested attribute
    def get_nested_attr(self, attr, *args):
        import functools
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [self] + attr.split('.'))
    
    


    ## BAYESIAN OPTIMIZATION

    def optimize(self, run_name=None, parameters=None, merit_fcn=None, label=None, num_shots_per_step=1, num_steps=50, savedepth=2, verbose=None, overwrite=False, parallel=False, max_cores=16):

        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "optimization_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        if overwrite:
            self.clear_run_data()
            overwrite = False
            
        # initialize the step
        self.step = 0
        
        # save the merit function
        self.merit_fcn = merit_fcn
        
        # perform run
        def evaluation_function(params):
            
            # set the parameters
            opt_fcn = lambda obj, _: obj.set_parameters(params)
            
            # run the simulations
            self.scan(run_name=self.run_name, fcn=opt_fcn, vals=np.arange(num_steps), step_filter=self.step, num_shots_per_step=num_shots_per_step, savedepth=savedepth, verbose=verbose, overwrite=overwrite, parallel=parallel, max_cores=max_cores, label=label)
            
            # evaluate the merit function
            vals = np.empty(self.num_shots_per_step)
            for shot_in_step in range(self.num_shots_per_step):
                vals[shot_in_step] = self.merit_fcn(self[self.step, shot_in_step])

            # take mean value
            val_mean = np.nanmean(vals)
            
            # iterate the optimzation step
            self.step += 1

            # print the merit function and parameters
            print(f"Merit function ({label}): {val_mean:.3g}")
            for key in params:
                print(f">> {key}: {params[key]:.3g}")
            
            return val_mean
        
        # perform optimization
        from ax import optimize as bayes_opt
        best_parameters, best_values, experiment, model = bayes_opt(
            parameters = parameters,
            evaluation_function = evaluation_function,
            minimize = True,
            total_trials = num_steps
        )
        
        return best_parameters, best_values
    
    
    # Extract mean and standard deviation value of beam parameters across a scan
    def extract_beam_function(self, beam_fcn, index=-1):
        val_mean, val_std = self.extract_function(lambda obj : beam_fcn(obj.get_beam(index=index)))
        return val_mean, val_std
            
    def extract_function(self, fcn):

        # prepare the arrays
        val_mean = np.empty(self.num_steps)
        val_std = np.empty(self.num_steps)
        
        # extract values
        if self.is_scan():
            
            for step in range(self.num_steps):
                
                # get values for this step
                val_output = np.empty(self.num_shots_per_step)
                for shot_in_step in range(self.num_shots_per_step):
                    val_output[shot_in_step] = fcn(self[step, shot_in_step])
                    
                # get step mean and error
                val_mean[step] = np.mean(val_output)
                val_std[step] = np.std(val_output)
        
        return val_mean, val_std


    
    ## PLOT FUNCTIONS

    # plot value of beam parameters across a scan
    def plot_function(self, fcns, label=None, scale=1, xscale='linear', yscale='linear', legend=None):

        from matplotlib import pyplot as plt
        
        if not isinstance(fcns, list):
            fcns = [fcns]
        
        if not isinstance(legend, list):
            legend = [legend]
    
        # plot evolution
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)
        
        for i, fcn in enumerate(fcns):
            
            # extract values
            val_mean, val_std = self.extract_function(fcn)
            
            if not hasattr(self, 'scale'):
                self.scale = 1
            if not hasattr(self, 'label'):
                self.label = ''
            
            ax.errorbar(self.vals/self.scale, val_mean/scale, abs(val_std/scale), ls=':', capsize=5, label=legend[i])
            if legend[i] is not None:
                ax.legend()
            
        ax.set_xlabel(self.label)
        ax.set_ylabel(label)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    
    # plot value of beam parameters across a scan
    def plot_beam_function(self, beam_fcn, index=-1, label=None, scale=1, xscale='linear', yscale='linear'):
        self.plot_function(lambda obj : beam_fcn(obj.get_beam(index=index)), label=label, scale=scale, xscale=xscale, yscale=yscale)

    def plot_energy(self, index=-1):
        self.plot_beam_function(Beam.energy, scale=1e9, label='Energy [GeV]', index=index)

    def plot_energy_spread(self, index=-1):
        self.plot_beam_function(Beam.rel_energy_spread, scale=1e-2, label='Energy spread, rms [%]', index=index)

    def plot_charge(self, index=-1):
        self.plot_beam_function(Beam.abs_charge, scale=1e-9, label='Charge [nC]', index=index)

    def plot_beam_size_x(self, index=-1):
        self.plot_beam_function(Beam.beam_size_x, scale=1e-3, label='Beam size, x [mm rms]', index=index)

    def plot_beam_size_y(self, index=-1):
        self.plot_beam_function(Beam.beam_size_y, scale=1e-3, label='Beam size, y [mm rms]', index=index)
    
    def plot_offset_x(self, index=-1):
        self.plot_beam_function(Beam.x_offset, scale=1e-3, label='Offset, x [mm]', index=index)

    def plot_offset_y(self, index=-1):
        self.plot_beam_function(Beam.y_offset, scale=1e-3, label='Offset, y [mm]', index=index)
        

    ## PLOT WATERFALLS
    
    def plot_waterfall(self, proj_fcn, label=None, scale=1, index=-1):

        from matplotlib import pyplot as plt
        
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
        self.plot_waterfall(Beam.energy_spectrum, scale=1e9, label='Energy [GeV]', index=index)

    def plot_waterfall_current(self, index=-1):
        self.plot_waterfall(Beam.current_profile, scale=1e-15, label='Time [fs]', index=index)

    def plot_waterfall_x(self, index=-1):
        self.plot_waterfall(Beam.transverse_profile_x, scale=1e-3, label='x [mm]', index=index)

    def plot_waterfall_y(self, index=-1):
        self.plot_waterfall(Beam.transverse_profile_y, scale=1e-3, label='y (mm)', index=index)


    ## PLOT CORRELATIONS
    
    def plot_correlation(self, xfcn, yfcn, xlabel=None, ylabel=None, xscaling=1, yscaling=1, xscale='linear', yscale='linear', equal_axes=False):

        from matplotlib import pyplot as plt
        import abel.utilities.colors as cmaps
        from matplotlib import colors
        from matplotlib import cm
        
        # extract values
        valx_mean, valx_std = self.extract_function(xfcn)
        valy_mean, valy_std = self.extract_function(yfcn)
        
        if not hasattr(self, 'scale'):
            self.scale = 1
        if not hasattr(self, 'label'):
            self.label = ''
            
        # plot evolution
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # create a scatter plot
        cmap = cmaps.FLASHForward_nowhite
        sc = ax.scatter(valx_mean,valy_mean, s=0, c=self.vals/self.scale, cmap=cmap)

        # create colorbar according to the scatter plot
        clb = fig.colorbar(sc)
        
        # convert scan value to a color tuple using the colormap used for scatter
        norm = colors.Normalize(vmin=min(self.vals/self.scale), vmax=max(self.vals/self.scale), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cols = np.array([(mapper.to_rgba(v)) for v in self.vals/self.scale])
    
        for x, y, ex, ey, color in zip(valx_mean/xscaling, valy_mean/yscaling, abs(valx_std/xscaling), abs(valy_std/yscaling), cols):
            ax.errorbar(x, y, xerr=ex, yerr=ey, capsize=5, color=color)
        clb.ax.set_ylabel(self.label)
        if equal_axes:
            ax.axis('equal')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        
    def plot_beam_correlation(self, beam_xfcn, beam_yfcn, index=-1, xlabel=None, ylabel=None, xscaling=1, yscaling=1, xscale='linear', yscale='linear', equal_axes=False):
        self.plot_correlation(lambda obj: beam_xfcn(obj.get_beam(index=index)), lambda obj: beam_yfcn(obj.get_beam(index=index)), xlabel=xlabel, ylabel=ylabel, xscaling=xscaling, yscaling=yscaling, xscale=xscale, yscale=yscale, equal_axes=equal_axes)

    def plot_correlation_offsets(self):
        self.plot_beam_correlation(Beam.x_offset, Beam.y_offset, xscaling=1e-3, yscaling=1e-3, xlabel='Offset, x (mm)', ylabel='Offset, y (mm)', equal_axes=True)

    
    ## SAVE TO FILE
    
    def save_function_data(self, fcn, filename=None):

        # extract mean and std values
        val_mean, val_std = self.extract_function(fcn)
        
        # default filename
        if filename is None:
            filename = 'output_' + self.run_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'

        # write data
        data = [self.vals, val_mean, val_std]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    