from abel import Stage, Beam, CONFIG
import scipy.constants as SI
from matplotlib import pyplot as plt
from abel.utilities.plasma_physics import blowout_radius, k_p, beta_matched
from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
import os, shutil, uuid
import numpy as np
import wake_t
from openpmd_viewer import OpenPMDTimeSeries

class StageWakeT(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source
        self.ramp_beta_mag = ramp_beta_mag


        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)

        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        
        # apply plasma-density up ramp (demagnify beta function)
        driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')
        
        # make longitudinal box range
        num_sigmas = 6
        box_min_z = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs())+0.25/k_p(self.plasma_density))
        box_range_z = [box_min_z, box_max_z]
        
        # making transverse box size
        box_size_r = np.max([5/k_p(self.plasma_density), 2*blowout_radius(self.plasma_density, driver0.peak_current())])
        
        # find stepsize
        beta_matched = np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))/k_p(self.plasma_density)
        dz = beta_matched/10
        
        n_out = round(self.length/dz/8)
        plasma = wake_t.PlasmaStage(length=self.length, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=128, n_xi=256, dz_fields=dz, ppc=1)
        
        # do tracking
        bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder)
        
        # save evolution of the beam and driver
        self.__extract_evolution(bunches)
        self.__extract_initial_and_final_step(tmpfolder)
        
        # remove temporary directory
        shutil.rmtree(tmpfolder)
        
        # extract beams
        beam = wake_t_bunch2beam(bunches[1][-1])
        driver = wake_t_bunch2beam(bunches[0][-1])
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # apply plasma-density down ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        driver.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # calculate efficiency
        self.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.calculate_beam_current(beam0, driver0, beam, driver)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def __extract_evolution(self, bunches):

        # get beam
        beam_evol = wake_t.diagnostics.analyze_bunch_list(bunches[1])
        
        # store variables
        self.evolution.location = beam_evol['prop_dist']
        self.evolution.charge = beam_evol['q_tot']
        self.evolution.energy = beam_evol['avg_ene']*SI.m_e*SI.c**2/SI.e
        #self.evolution.z = beam_evol['z_avg']
        self.evolution.x = beam_evol['x_avg']
        self.evolution.y = beam_evol['y_avg']
        self.evolution.xp = beam_evol['theta_x']
        self.evolution.yp = beam_evol['theta_y']
        self.evolution.rel_energy_spread = beam_evol['rel_ene_spread']
        self.evolution.energy_spread = self.evolution.rel_energy_spread*self.evolution.energy
        self.evolution.beam_size_x = beam_evol['sigma_x']
        self.evolution.beam_size_y = beam_evol['sigma_y']
        self.evolution.bunch_length = beam_evol['sigma_z']
        self.evolution.emit_nx = beam_evol['emitt_x']
        self.evolution.emit_ny = beam_evol['emitt_y']
        self.evolution.beta_x = beam_evol['beta_x']
        self.evolution.beta_y = beam_evol['beta_y']

    
    def __extract_initial_and_final_step(self, tmpfolder):
        
        # prepare to read simulation data
        source_path = tmpfolder + 'hdf5/'
        ts = OpenPMDTimeSeries(source_path)

        # extract initial on-axis wakefield
        Ez0, metadata0 = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=min(ts.iterations))
        self.initial.plasma.wakefield.onaxis.zs = metadata0.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez0
        
        # extract final on-axis wakefield
        Ez, metadata = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=max(ts.iterations))
        self.final.plasma.wakefield.onaxis.zs = metadata.z
        self.final.plasma.wakefield.onaxis.Ezs = Ez
        
        # extract initial plasma density
        rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
        self.initial.plasma.density.extent = metadata0_plasma.imshow_extent
        self.initial.plasma.density.rho = -(rho0_plasma/SI.e)
        
        # extract final beam density
        data0_beam = ts.get_particle(species='beam', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        data0_driver = ts.get_particle(species='driver', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        extent0 = metadata0_plasma.imshow_extent
        Nbins0 = self.initial.plasma.density.rho.shape
        dr0 = (extent0[3]-extent0[2])/Nbins0[0]
        dz0 = (extent0[1]-extent0[0])/Nbins0[1]
        mask0_beam = np.logical_and(data0_beam[1] < dr0/2, data0_beam[1] > -dr0/2)
        jz0_beam, _, _ = np.histogram2d(data0_beam[0][mask0_beam], data0_beam[2][mask0_beam], weights=data0_beam[3][mask0_beam], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        mask0_driver = np.logical_and(data0_driver[1] < dr0/2, data0_driver[1] > -dr0/2)
        jz0_driver, _, _ = np.histogram2d(data0_driver[0][mask0_driver], data0_driver[2][mask0_driver], weights=data0_driver[3][mask0_driver], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        self.initial.beam.density.extent = metadata0_plasma.imshow_extent
        self.initial.beam.density.rho = (jz0_beam+jz0_driver)/(dr0*dr0*dz0)

        # extract initial plasma density
        rho_plasma, metadata_plasma = ts.get_field(field='rho', iteration=max(ts.iterations))
        self.final.plasma.density.extent = metadata_plasma.imshow_extent
        self.final.plasma.density.rho = -(rho_plasma/SI.e)

        # extract final beam density
        data_beam = ts.get_particle(species='beam', var_list=['x','y','z','w'], iteration=max(ts.iterations))
        data_driver = ts.get_particle(species='driver', var_list=['x','y','z','w'], iteration=max(ts.iterations))
        extent = metadata_plasma.imshow_extent
        Nbins = self.final.plasma.density.rho.shape
        dr = (extent[3]-extent[2])/Nbins[0]
        dz = (extent[1]-extent[0])/Nbins[1]
        mask_beam = np.logical_and(data_beam[1] < dr/2, data_beam[1] > -dr/2)
        jz_beam, _, _ = np.histogram2d(data_beam[0][mask_beam], data_beam[2][mask_beam], weights=data_beam[3][mask_beam], bins=Nbins, range=[extent[2:4],extent[0:2]])
        mask_driver = np.logical_and(data_driver[1] < dr/2, data_driver[1] > -dr/2)
        jz_driver, _, _ = np.histogram2d(data_driver[0][mask_driver], data_driver[2][mask_driver], weights=data_driver[3][mask_driver], bins=Nbins, range=[extent[2:4],extent[0:2]])
        self.final.beam.density.extent = metadata_plasma.imshow_extent
        self.final.beam.density.rho = (jz_beam+jz_driver)/(dr*dr*dz)
        
        
    def energy_usage(self):
        return None # TODO
    
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
        