from abel import Spectrometer
import numpy as np
import scipy.constants as SI
import scipy
import ocelot
import ocelot.gui.accelerator as ocelot_gui
from abel.apis.ocelot.ocelot_api import ocelot_particle_array2beam, beam2ocelot_particle_array

class SpectrometerFacetOcelot(Spectrometer):
    
    def __init__(self, bend_angle=-0.03, img_energy=None, obj_plane=0, mag_x=-4, img_energy_y=None, obj_plane_y=None, exact_tracking=True):   
        
        self.bend_angle = bend_angle # [rad]
        self.img_energy = img_energy
        self.obj_plane = obj_plane
        self.mag_x = mag_x
        self.img_energy_y = img_energy_y
        self.obj_plane_y = obj_plane_y
        self.exact_tracking = exact_tracking
        
        # initial guess for quadrupole strengths
        self.ks = [-0.41076614, 0.57339363, -0.36132452] # [m^-2]
        
    
    # lattice length
    def get_length(self):
        return self.get_lattice().totalLen
    
    
    # get the Ocelot lattice
    def get_lattice(self, ks=None, obj_plane=None):

        # default values if not defined
        if ks is None:
            ks = self.ks
        if obj_plane is None:
            obj_plane = 0
            
        # define elements
        drift0 = ocelot.Drift(l=1.897-obj_plane)
        quad_Q0 = ocelot.Quadrupole(l=1.0, k1=ks[0])
        drift1 = ocelot.Drift(l=1.224)
        quad_Q1 = ocelot.Quadrupole(l=1.0, k1=ks[1])
        drift2 = ocelot.Drift(l=1.224)
        quad_Q2 = ocelot.Quadrupole(l=1.0, k1=ks[2])
        drift3 = ocelot.Drift(l=3.520)
        dipole = ocelot.RBend(l=0.978, angle=self.bend_angle, tilt=-np.pi/2)
        drift4 = ocelot.Drift(l=8.831)

        # assemble element sequence
        sequence = (drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4)

        # select tracking method (second-order matrices or exact kicks)
        if self.exact_tracking:
            tracking_method = ocelot.KickTM
        else:
            tracking_method = ocelot.SecondTM
        
        # define lattice
        lattice = ocelot.MagneticLattice(sequence, method={'global': tracking_method})
        
        return lattice

    
    # set the quad strengths for imaging
    def set_imaging(self, nom_energy=None): 
        
        # TODO: check if the object plane is set correctly
        
        def img_condition_fcn(ks):
            
            # set vertical plane if not already
            if self.img_energy_y is None:
                self.img_energy_y = self.img_energy
            if self.obj_plane_y is None:
                self.obj_plane_y = self.obj_plane
            
            # calculate transfer matrices
            lattice_x = self.get_lattice(ks, self.obj_plane)
            Rx = ocelot.lattice_transfer_map(lattice_x, energy=self.img_energy/1e9)
            lattice_y = self.get_lattice(ks*self.img_energy_y/self.img_energy, self.obj_plane_y)
            Ry = ocelot.lattice_transfer_map(lattice_y, energy=self.img_energy/1e9)

            # return object function
            return (Rx[0,1])**2 + (Ry[2,3])**2 + (Rx[0,0]-self.mag_x)**2

        # perform minization (find k-values)
        result = scipy.optimize.minimize(img_condition_fcn, self.ks, tol=1e-5, options={'maxiter': 1000})
        
        # set solution to quads
        if result.fun < 1e-5:
            
            # scale quadrupole strengths to the nominal energy
            if nom_energy is None:
                self.ks = result.x
            else:
                self.ks = result.x * self.img_energy/nom_energy
                
        else:
            raise Exception('No imaging solution found.')
           
    
    # tracking function
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # set imaging
        self.set_imaging(nom_energy=beam0.energy())
        
        # convert beam to Ocelot particle array
        p_array0 = beam2ocelot_particle_array(beam0)
        
        # get lattice (
        lattice = self.get_lattice(obj_plane=0)
        
        # perform tracking
        _, p_array = ocelot.track(lattice, p_array0, navi=ocelot.Navigator(lattice), print_progress=False)
        
        # calculate Twiss evolution
        twiss0 = ocelot.get_envelope(p_array0)
        twiss_list = ocelot.twiss(lattice, twiss0)
        
        # convert back from Ocelot particle array
        beam = ocelot_particle_array2beam(p_array)
        
        # shift beam by the dispersion (special case for spectrometers)
        beam.set_ys(beam.ys() - twiss_list[-1].Dy)
        
        # re-add metadata (before iterating)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)
        
    
    # plot the evolution of beta functions and dispersion
    def plot_twiss(self, energy=None, waist_plane=None):
        
        # set the imaging
        self.set_imaging(nom_energy=energy)
        
        # get lattice
        if waist_plane is None:
            waist_plane = 0
        lattice = self.get_lattice(obj_plane=waist_plane)

        # example Twiss (small beta functions)
        twiss0 = ocelot.Twiss()
        twiss0.beta_x = 0.05
        twiss0.alpha_x = 0
        twiss0.beta_y = 0.05
        twiss0.alpha_y = 0

        # calculate Twiss evolution
        twiss = ocelot.twiss(lattice, twiss0, nPoints=100)
        
        # plot evolution
        ocelot_gui.plot_opt_func(lattice, twiss, top_plot=['Dy'], legend=False)
        