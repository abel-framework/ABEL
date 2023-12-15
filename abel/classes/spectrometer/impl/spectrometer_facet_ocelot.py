from abel import Spectrometer
import numpy as np
import scipy.constants as SI
import scipy
from abel.apis.ocelot.ocelot_api import ocelot_particle_array2beam, beam2ocelot_particle_array


class SpectrometerFacetOcelot(Spectrometer):
    
    def __init__(self, dipole_field=-1, img_energy=None, obj_plane=0, mag_x=-4, img_energy_y=None, obj_plane_y=None, exact_tracking=True):   
        
        self.dipole_field = dipole_field
        self.img_energy = img_energy
        self.obj_plane = obj_plane
        self.mag_x = mag_x
        self.img_energy_y = img_energy_y
        self.obj_plane_y = obj_plane_y
        self.exact_tracking = exact_tracking
        
        # initial guess for quadrupole strengths
        self.ks_for_img_energy = [-0.41076614, 0.57339363, -0.36132452] # [m^-2]
        
    
    # lattice length
    def get_length(self):
        return self.get_lattice().totalLen
    
    
    # get the Ocelot lattice
    def get_lattice(self, ks=None, energy=None, obj_plane=None):
        
        # import OCELOT
        from ocelot import Drift, Quadrupole, RBend, KickTM, SecondTM, MagneticLattice
        
        # dipole length
        dipole_length = 0.978
        
        # default energy if not defined
        if energy is None:
            energy = self.img_energy
        if obj_plane is None:
            obj_plane = 0
        
        # scale quad and dipole strengths to correct energy
        if ks is None:
            ks = self.ks_for_img_energy*self.img_energy/energy        
        bend_angle = self.dipole_field*dipole_length*SI.c/energy

        # define elements
        drift0 = Drift(l=1.897-obj_plane)
        quad_Q0 = Quadrupole(l=1.0, k1=ks[0])
        drift1 = Drift(l=1.224)
        quad_Q1 = Quadrupole(l=1.0, k1=ks[1])
        drift2 = Drift(l=1.224)
        quad_Q2 = Quadrupole(l=1.0, k1=ks[2])
        drift3 = Drift(l=3.520)
        dipole = RBend(l=dipole_length, angle=bend_angle, tilt=-np.pi/2)
        drift4 = Drift(l=8.831)

        # assemble element sequence
        sequence = (drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4)

        # select tracking method (second-order matrices or exact kicks)
        if self.exact_tracking:
            tracking_method = KickTM
        else:
            tracking_method = SecondTM
        
        # define lattice
        lattice = MagneticLattice(sequence, method={'global': tracking_method})
        
        return lattice

    
    # set the quad strengths for imaging
    def set_imaging(self): 
        
        # import OCELOT
        from  ocelot import lattice_transfer_map
        
        # TODO: check if the object plane is set correctly
        
        def img_condition_fcn(ks):
            
            # set vertical plane if not already
            if self.img_energy_y is None:
                self.img_energy_y = self.img_energy
            if self.obj_plane_y is None:
                self.obj_plane_y = self.obj_plane
            
            # calculate transfer matrices
            
            lattice_x = self.get_lattice(ks, self.img_energy, self.obj_plane)
            Rx = lattice_transfer_map(lattice_x, energy=self.img_energy/1e9)
            lattice_y = self.get_lattice(ks, self.img_energy_y, self.obj_plane_y)
            Ry = lattice_transfer_map(lattice_y, energy=self.img_energy/1e9)

            # return object function
            return (Rx[0,1])**2 + (Ry[2,3])**2 + (Rx[0,0]-self.mag_x)**2

        # perform minization (find k-values)
        result = scipy.optimize.minimize(img_condition_fcn, self.ks_for_img_energy, tol=1e-5, options={'maxiter': 1000})
        
        # set solution to quads
        if result.fun < 1e-5:
            
            # scale quadrupole strengths to the nominal energy
            self.ks_for_img_energy = result.x
                
        else:
            raise Exception('No imaging solution found.')
           
    
    # tracking function
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # import OCELOT
        from ocelot import track, twiss, get_envelope
        
        # set imaging
        self.set_imaging()
        
        # convert beam to Ocelot particle array
        p_array0 = beam2ocelot_particle_array(beam0)
        
        # get lattice (
        lattice = self.get_lattice(energy=beam0.energy(), obj_plane=0)
        
        # perform tracking
        _, p_array = track(lattice, p_array0, navi=ocelot.Navigator(lattice), print_progress=False)
        
        # calculate Twiss evolution
        twiss0 = get_envelope(p_array0)
        twiss_list = twiss(lattice, twiss0)
        
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
        
        # import OCELOT
        from ocelot import Twiss, twiss
        from ocelot.gui.accelerator import plot_opt_func
        
        # set the imaging
        self.set_imaging()
        
        # get lattice
        if energy is None:
            energy = self.img_energy
        if waist_plane is None:
            waist_plane = 0
        lattice = self.get_lattice(energy=energy, obj_plane=waist_plane)

        # example Twiss (small beta functions)
        twiss0 = Twiss()
        twiss0.beta_x = 0.05
        twiss0.alpha_x = 0
        twiss0.beta_y = 0.05
        twiss0.alpha_y = 0
        
        # calculate Twiss evolution
        twiss_evol = twiss(lattice, twiss0, nPoints=100)
        
        # plot evolution
        plot_opt_func(lattice, twiss_evol, top_plot=['Dy'], legend=False)

    
    def get_dispersion(self, energy=None):
        
        # import OCELOT
        from ocelot import Twiss, twiss
        
        # set default energy
        if energy is None:
            energy = self.img_energy
            
        # set the imaging and get the lattice
        self.set_imaging()
        lattice = self.get_lattice(energy=energy)
        
        # calculate Twiss evolution
        twiss0 = Twiss()
        twiss0.beta_x = 0.05
        twiss0.alpha_x = 0
        twiss0.beta_y = 0.05
        twiss0.alpha_y = 0
        twiss_evol = twiss(lattice, twiss0)
        
        # extract dispersion
        dispersion = twiss_evol[-1].Dy
        
        return dispersion

    
    def get_m12(self, energies):
        
        # import OCELOT
        from ocelot import lattice_transfer_map
        
        m12s = np.zeros(len(energies))
        for i in range(len(energies)):
            lattice_x = self.get_lattice(energy=energies[i], obj_plane=self.obj_plane)
            Rx = lattice_transfer_map(lattice_x, energy=energies[i]/1e9)
            m12s[i] = Rx[0, 1]
        
        return m12s
    