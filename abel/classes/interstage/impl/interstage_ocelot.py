import scipy.constants as SI
import numpy as np
from abel import Interstage
#import ocelot
#import ocelot.gui.accelerator as ocelot_gui
from abel.apis.ocelot.ocelot_api import ocelot_particle_array2beam, beam2ocelot_particle_array

class InterstageOcelot(Interstage):
    
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None, beta0=None):
        self.nom_energy = nom_energy
        self.dipole_length = dipole_length
        self.dipole_field = dipole_field
        self.beta0 = beta0
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # import OCELOT
        import ocelot
        
        # convert beam to Ocelot particle array
        p_array0 = beam2ocelot_particle_array(beam0)
        
        # get lattice
        lattice = self.get_lattice()
        
        # make navigator
        navigator = ocelot.Navigator(lattice)
        
        # perform tracking
        _, p_array = ocelot.track(lattice, p_array0, navi=navigator, print_progress=False)
        
        # calculate Twiss evolution
        self.__twiss0 = ocelot.get_envelope(p_array0)
        twiss_list = ocelot.twiss(lattice, self.__twiss0)
        
        # convert back from Ocelot particle array
        beam = ocelot_particle_array2beam(p_array)
        
        # re-add metadata (before iterating)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # plot the evolution of beta functions and dispersion
    def plot_twiss(self):
        
        # import OCELOT
        import ocelot
        import ocelot.gui.accelerator as ocelot_gui
        
        # extract the Twiss parameters if they exist
        twiss0 = ocelot.Twiss()
        twiss0.beta_x = self.beta0
        twiss0.alpha_x = 0
        twiss0.beta_y = self.beta0
        twiss0.alpha_y = 0
            
        # get lattice
        lattice = self.get_lattice()

        # calculate Twiss evolution
        twiss = ocelot.twiss(lattice, twiss0, nPoints=100)
        
        # plot evolution
        ocelot_gui.plot_opt_func(lattice, twiss, top_plot=['Dx'], legend=False)
     
    
    # evaluate longitudinal dispersion (R56)
    def R_56(self):
        return -self.dipole_field**2*SI.c**2*self.__eval_dipole_length()**3/(3*self.nom_energy**2)
    
       
    # get the Ocelot lattice
    def get_lattice(self):
        
        # import OCELOT
        import ocelot
        
        if callable(self.dipole_length):
            l_dipole1 = self.dipole_length(self.nom_energy)
        else:
            l_dipole1 = self.dipole_length
        
        # define spacer drift
        spacer = ocelot.Drift(l=0.05*l_dipole1)
        
        # define main dipole
        theta_dip = self.dipole_field*l_dipole1*SI.c/self.nom_energy # [rad]
        dipole1 = ocelot.SBend(l=l_dipole1, angle=theta_dip)
        
        # define plasma lens
        g_max = 1000 # [T/m]
        l_plasmalens = 2*self.nom_energy/(l_dipole1*g_max*SI.c)
        k_guess = 2*l_dipole1/l_plasmalens
        #plasmalens = ocelot.FieldMap(field_file=)
        plasmalens_fake = ocelot.Quadrupole(l=l_plasmalens, k1=k_guess)

        # define second and third dipoles
        l_dipole23 = 0.6 * l_dipole1
        dipole2 = ocelot.SBend(l=l_dipole23, angle=theta_dip*0.8)
        dipole3 = ocelot.SBend(l=l_dipole23, angle=-theta_dip*0.8)

        # define sextupole
        l_sextupole = 0.4 * l_dipole1
        sextupole_half = ocelot.Sextupole(l=l_sextupole/2, k2=0)
        
        # add midpoint monitor
        midpoint = ocelot.Monitor()
        endpoint = ocelot.Monitor()

        # define lattice
        half_seq = (spacer, dipole1, spacer, plasmalens_fake, spacer, dipole2, spacer, dipole3, spacer, sextupole_half, midpoint)
        seq = half_seq + (midpoint,)
        lattice_fake = ocelot.MagneticLattice(seq)
        
        # perform matching
        twiss0 = ocelot.Twiss()
        twiss0.beta_x = self.beta0
        twiss0.alpha_x = 0
        twiss0.beta_y = self.beta0
        twiss0.alpha_y = 0

        constr = {midpoint:{'alpha_x':0, 'Dxp':0}}
        vars = [plasmalens_fake, dipole2]
        ocelot.match(lattice_fake, constr, vars, twiss0, verbose=True, method='simplex', vary_bend_angle=True)
        k = plasmalens_fake.k1
        print(plasmalens_fake)
        print(dipole2)
        
        # re-define lattice
        l = l_plasmalens
        plasmalens = ocelot.Matrix(l=l_plasmalens, 
                         rm11=np.cos(np.sqrt(k)*l), rm12=np.sin(np.sqrt(k)*l)/np.sqrt(k),
                         rm21=-np.sin(np.sqrt(k)*l)*np.sqrt(k), rm22=np.cos(np.sqrt(k)*l),
                         rm33=np.cos(np.sqrt(k)*l), rm34=np.sin(np.sqrt(k)*l)/np.sqrt(k),
                         rm43=-np.sin(np.sqrt(k)*l)*np.sqrt(k), rm44=np.cos(np.sqrt(k)*l))

        half_seq = (spacer, dipole1, spacer, plasmalens, spacer, dipole2, spacer, dipole3, spacer, sextupole_half)
        seq = half_seq + (midpoint,)
        for element in reversed(half_seq):
            seq += (element,)
        lattice = ocelot.MagneticLattice(seq)

        #R = ocelot.lattice_transfer_map(lattice, energy=self.nom_energy/1e9)
        _, R, T = lattice.transfer_maps(energy=self.nom_energy/1e9)
        print(R)
        print(T)
        
        return lattice


    
    # lattice length
    def get_length(self):
        return self.get_lattice().totalLen
     
        