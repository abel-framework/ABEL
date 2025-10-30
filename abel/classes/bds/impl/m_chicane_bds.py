import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
from abel.classes.source.impl.source_basic import SourceBasic
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint


class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=1, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False, x0=10, lattice=[], ns=100, beta0_x=1, beta0_y=1,\
                    alpha0_x=0, alpha0_y=0, Dx0=0, Dpx0=0, R560=0, k_bound=10):
        # E_nom in eV, delay in ns
        super().__init__()
        self._E_nom = E_nom #backing variable 
        self._delay_per_stage = delay_per_stage*1e-9
        self.num_stages = num_stages
        self._length_stage = length_stage
        self.keep_data = keep_data
        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.ks = ks
        self.layoutnr = layoutnr
        self.use_monitors = use_monitors
        self.x0 = x0
        self.lattice = lattice
        self.ns = ns
        self.beta0_x = beta0_x
        self.beta0_y = beta0_y
        self.alpha0_x = alpha0_x
        self.alpha0_y = alpha0_y
        self.Dx0 = Dx0
        self.Dpx0 = Dpx0
        self.R560 = R560
        self.k_bound = k_bound

        # Lattice-elements lengths
        self.l_quads = 0.3
        self.l_kick = 2
        self.l_gap = 1
        self.l_diag = l_diag
        self.l_dipole = self.get_dipole_lengths()
        self.l_sext = 0.1
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        self.B_dipole = B_dipole
        self.full_extend=0

    @property
    def E_nom(self):
        return self._E_nom
    
    @property
    def delay_per_stage(self):
        return self._delay_per_stage
    
    @property
    def length_stage(self):
        return self._length_stage
    
    @length_stage.setter
    def length_stage(self, value):
        self._length_stage = value

    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value*1e-9


    # Setting the new E_nom
    
    @E_nom.setter
    def E_nom(self, value):
        self._E_nom = value

    def update_lattice(self):
        # Get the indices of the quads in the lattice
        self.indices_quads = [i for i, element in enumerate(self.lattice) if element.name=='quad']
        self.indices_dipoles = [i for i, element in enumerate(self.lattice) if element.name=='dipole']

    def scale_lattice(self, scale=1):
        for element in self.lattice:
            element.ds *=scale

    ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
    # If I find the root of this function (I.e where it is 0, as it should be) by 
    def constraint(self, B_multiplier=1, plot=False, rotate=False):
        from abel.utilities.beam_physics import evolve_orbit
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        """
        params= [k1, k2, k3, k4, B1, B2, B3], the B-fields to be optimized. Same input as in the optimize-function
        """

        ls, inv_rs = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos')) # dipoles = (bool, 'phi' OR 'inv_rhos'))
        inv_rs = [inv_r*B_multiplier for inv_r in inv_rs]
        
        theta, evolution = evolve_orbit(ls=ls, inv_rhos=inv_rs, plot=plot) # evolution[0,:] = xs is the projected length

        hypotenuse = np.sqrt(evolution[0,-1]**2 + evolution[1,-1]**2)
        if rotate:
            phi = np.arctan(evolution[1,-1]/evolution[0,-1])
        else:
            phi = 0
        print("phi0 = ", phi, 'phi_end = ', theta)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        elements_location = self.get_z_elements(theta0=-phi)
        dipoles = elements_location['dipoles']
        quads = elements_location['quads']

        rotated_data = np.dot(np.array([evolution[0,:], evolution[1,:]]).T, R)
        rotated_x = rotated_data[:, 0]
        rotated_y = rotated_data[:, 1]
        fig, ax = plt.subplots(2,1,figsize=(8,3), gridspec_kw={'height_ratios': [4, 1]})
        fig.tight_layout()
        
        ax[0].plot(rotated_x, rotated_y)
        ax[0].grid()
        ax[0].set_xlabel("z (m)")
        ax[0].set_ylabel("x (m)")
        ax[0].set_title("Delay chicane orbit")
        ax[1].tick_params(axis='y', left=False, labelleft=False)
        ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
        # Define magnet dimensions
        dipole_width = 0.8
        dipole_height = 0.2 
        quad_width = 0.15
        quad_height = 0.4 # Taller than dipoles

        # Plot dipoles (wide, short rectangles)
        for z in dipoles:
            # The rectangle is centered at (z, 0)
            z_ = z[0]
            pos = z[1]
            corner_x = z_ - dipole_width / 2
            if pos==1:
                corner_y = 0
            elif pos==-1:
                corner_y = -dipole_height
            ax[1].add_patch(patches.Rectangle((corner_x, corner_y), dipole_width, dipole_height, facecolor='black', label='Dipole'))

        # Plot quadrupoles (thin, tall rectangles)
        for z in quads:
            # The x-coordinate calculation is the same for all quads
            z_ = z[0]
            pos = z[1]
            corner_x = z_ - quad_width / 2
            
            # Check if the quad is at an even (0, 2, 4...) or odd (1, 3, 5...) position
            if pos == 1:
                # EVEN: Center the rectangle on the y=0 axis
                corner_y = 0
            elif pos == -1:
                # ODD: Place the bottom of the rectangle on the y=0 axis, extending downwards
                corner_y = -quad_height
                
            ax[1].add_patch(patches.Rectangle((corner_x, corner_y), quad_width, quad_height, facecolor='black', label='Quadrupole'))
        
        # Auto-adjust the plot limits to fit the magnets and align axes
        ax[1].set_ylim(-(quad_height+0.5*quad_height), quad_height+0.5*quad_height) # Set y-limits based on tallest element
        ax[1].set_xlim(ax[0].get_xlim()) # Ensure x-axes are aligned
        ax[1].axhline(0, color='black', linewidth=1.2, zorder=0)
    
        #ax[1].spines['top'].set_visible(False)
        #ax[1].spines['right'].set_visible(False)
        #ax[1].spines['left'].set_visible(False)
        #ax[1].spines['bottom'].set_visible(False)       
        ax[1].set_aspect('auto', adjustable='box')


        plt.show()

        print('delay: ', (np.sum(ls)-evolution[0,-1])/SI.c*1e9)

        return evolution[0,-1] - self.length_stage, hypotenuse
    
    def get_z_elements(self, theta0=0):
        ls, phis, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'phis'), quads=True)

        elements_location={}
        dipoles = []
        quads = []
        L_z0 = 0

        for l, phi, k in zip(ls, phis, ks):
            if phi != 0:
                r = l/phi
                L_z = 2*abs(r)*np.sin(abs(phi)/2)*np.cos(theta0 + phi/2)
                
                if phi<0:
                    dipoles.append((L_z0 + L_z/2, -1))
                else:
                    dipoles.append((L_z0 + L_z/2, 1))

                theta0 += phi
                L_z0 += L_z
            elif k!=0:
                L_z = l*np.cos(theta0)
                if k<0:
                    quads.append((L_z0 + L_z/2, -1))
                else:
                    quads.append((L_z0 + L_z/2, 1))
                L_z0 += L_z
            else:
                L_z0 += l*np.cos(theta0)
                
        elements_location['dipoles'] = dipoles
        elements_location['quads'] = quads

        return elements_location




    def get_dipole_lengths(self):
        """
        Sets the lengths of the dipoles such that the delay is achieved
        """
        L = (self.length_stage + SI.c*self.delay_per_stage - self.l_gap/2 - self.l_kick/2 - self.l_diag)/4

        return L
    def get_quad_counts(self):
        N = sum(1 for element in self.lattice if element.name=='quad') - self.get_n_dobble_quads()
        return N
    
    def get_n_dobble_quads(self):
        """
        Check how many quads consists of two halfs that are to be identical
        """
        S=0
        for i in self.indices_quads:
            if i+1 in self.indices_quads:
                S+=1
        return S
    
    def get_k0_and_bounds(self):
        n_quads = self.get_quad_counts()

        k_bound = self.k_bound
        f0 = (self.length_stage+self.delay_per_stage*SI.c)/2
        if n_quads%2==0:
            k0 = [1/f0/self.l_quads, -1/f0/self.l_quads]*(n_quads//2)
            k_bounds = [(-k_bound,k_bound), (-k_bound,k_bound)]*(n_quads//2)
            print(len(k_bounds))
        else:
            k0 = [-1/f0/self.l_quads, 1/f0/self.l_quads]*(n_quads//2)
            k0.append(1/f0/self.l_quads)
            #k_bounds = [(-k_bound,0), (0,k_bound)]*(n_quads//2)
            k_bounds = [(0,k_bound), (-k_bound,0)]
            """
            ,(0,k_bound), (0,k_bound),(-k_bound,0)]
                      
            , (0,k_bound), (0,k_bound), (-k_bound,0), (0,k_bound)]
            """
            #k_bounds.append((-k_bound,0))

        return k0, k_bounds
    
    def match_dogleg(self, ks_matched=False, symmetric=True, mid=False):
        ### Optimize for alpha_x/y to get quad-strengths ###
        # ls_B = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56
        from scipy.optimize import minimize
        """
        Implement method of using the end-values for beta/alpha/Dispersion/R56 to
        match the second half of the stage. In this matching, ensure that beta->beta0 alpha=0, D=0,
        Dp=0, R56=0

        """
        p = self.E_nom/SI.c

        n_quads = self.get_quad_counts()
        print(n_quads)

        beta0_x = self.beta0_x
        beta0_y = self.beta0_y

        k_bound = 15
        if not ks_matched:
            k0, k_bounds = self.get_k0_and_bounds()
        else:
            k0 = [element.k for element in self.lattice if element.name=='quad']
            if n_quads%2==0:
                k_bounds = [(-k_bound, k_bound), (-k_bound, k_bound)]*(n_quads//2)
            else:
                k_bounds = [(-k_bound,k_bound), (-k_bound,k_bound)]*(n_quads//2)
                k_bounds.append((-k_bound,k_bound))

        ls, inv_rs_list, ks_list = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True) # returns numpy arrays

        
        def optimizer(params):
            #params = [k1, k2, ... k_-1]
            # Optimize from start (before dipole), but mirror the part from the dipole. The mirroring can be done
            # post optimizing 

            # Set the params values to the quads
            j=0
            for i in self.indices_quads:
                if i-1 in self.indices_quads: # There are 2 quads in a row
                    end_mid_mark = i # mark the index for the end of the first half
                    ks_list[i] = params[j-1] # make sure the next quad is identical
                else:
                    ks_list[i] = params[j]
                    j+=1    
                    

            # Evolve beam to mid (to element end_mid_mark)
            ls_1st_half = ls[:i]
            ks_1st_half = ks_list[:i]
            inv_rs_list_1st_half = inv_rs_list[:i]

            D_mid, _, _ = evolve_dispersion(ls=ls_1st_half, ks=ks_1st_half, inv_rhos=inv_rs_list_1st_half)
            _, alphax_mid, _ = evolve_beta_function(ls=ls_1st_half, ks=ks_1st_half, beta0=beta0_x)
            _, alphay_mid, _ = evolve_beta_function(ls=ls_1st_half, ks=-ks_1st_half, beta0=beta0_y)

            # Evolve beam entire way
            betax, alphax, _ = evolve_beta_function(ls=ls, ks=ks_list, beta0=beta0_x)
            betay, alphay, _ = evolve_beta_function(ls=ls, ks=-ks_list, beta0=beta0_y)
            D, Dp, _ = evolve_dispersion(ls=ls, ks=ks_list, inv_rhos=inv_rs_list)
            R56, _ = evolve_R56(ls=ls, ks=ks_list, inv_rhos=inv_rs_list)

            if mid:
                return (D*1e3)**2 #+ alphay_mid**2 + alphax_mid**2
            else:
                return (D*1e2)**2 + (D_mid*1e2)**2 + alphax**2 + alphay**2 \
                    + alphay_mid**2 + alphax_mid**2 # minimize dispersion, dispersion 
        
        opt = minimize(fun=optimizer, x0=k0, bounds=k_bounds)
        print(opt)
        ks = opt.x
        return ks

    def reset_lattice(self):
        try:
            self.lattice = self.lattice_backup
            self.update_lattice()
            self.set_ks(self.ks)
            self.lattice[self.indices_dipoles[0]].phi = self.lattice[self.indices_dipoles[0]].ds*self.inv_r
        except:
            print('No ks, inv_r or lattice_backed up yet')
        
            
    
    def set_ks(self, ks, lattice=None):
        j=0
        print(j)
        print(self.get_quad_counts())
        if not lattice:
            for i in self.indices_quads:
                if i-1 in self.indices_quads:
                    self.lattice[i].k = self.lattice[i-1].k
                else:
                    self.lattice[i].k = ks[j]
                    j+=1
        else:
            for element in lattice:
                if element.name == 'quad':
                    #print(ks[j])
                    element.k = ks[j]
                    j+=1

    def get_elements_arrays_from_lattice(self, lengths: bool=True, dipoles: tuple=(False, 'inv_rhos'), quads: bool=False, sext: bool=False):
        # self.lattice = {'dipole/quad': (length, magnetic variable), 'drift': length, ...}
        # Make the lists that are required for the evolution functions in beam_physics.py
        # self.lattice consists of impactx elements
        if lengths:
            ls = [element.ds for element in self.lattice]
            yield np.array(ls)

        if dipoles[0]:
            if dipoles[1]=='inv_rhos':
                inv_rhos = [element.phi/element.ds if hasattr(element, 'phi') else 0 for element in self.lattice]
                yield np.array(inv_rhos)
            elif dipoles[1]=='phis':
                phis = [element.phi if hasattr(element, 'phi') else 0 for element in self.lattice]
                yield np.array(phis)
            else:
                raise ValueError('must use ´inv_rhos´ or ´phi´')

        if quads:
            ks = [element.k if hasattr(element, 'k') and element.name=='quad' else 0 for element in self.lattice]
            yield np.array(ks)

        if sext:
            ms = [element.k_normal[2] if hasattr(element, 'k') and element.name=='sextupole' else 0 for element in self.lattice]
            yield ms

    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False, plot=False):

        from abel.apis.impactx.impactx_api import run_impactx
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56, evolve_second_order_dispersion

        # Get lattice
        #if not list(self.ks):
        #    print("matching quads")
        #    self.match_quads()
        
        #self.lattice = self.get_lattice(self.ks)
        #print(self.lattice)
        """
        Evolve beta function here
        """
        
        ls, inverse_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, "inv_rhos"), quads=True) # phis given in rads
        
        ms = [0]*len(ls)
        taus = [0]*len(ls)
        
        _, _, evox = evolve_beta_function(ls, ks, beta0=beam0.beta_x(), inv_rhos=inverse_rs, alpha0=beam0.alpha_x(), plot=plot)
        _, _, evoy = evolve_beta_function(ls, -np.array(ks), beta0=beam0.beta_y(), alpha0=beam0.alpha_y(), plot=plot)
        _, _, evoD = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, high_res=True, plot=plot)
        _, _, evoDD = evolve_second_order_dispersion(np.array(ls), inv_rhos=np.array(inverse_rs), ks=np.array(ks),\
                                                        ms=ms, taus=taus, plot=plot)
        _, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, high_res=True, plot=plot)

        # run ImpactX
        beam, evol = run_impactx(self.lattice, beam0, verbose=False, runnable=runnable, keep_data=self.use_monitors,\
                                  space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return sum(element.ds for element in self.lattice)
    
    def get_nom_energy(self):
        return self.E_nom
    
    
    def extend_lattice(self, full=False, quad_last=False):
        deep_copy = self.copy_lattice()
        if quad_last:
            deep_copy = deep_copy[:-1]

        if full and self.full_extend<1:
            for element in deep_copy:
                if element.name == 'dipole':
                    element.phi = -element.phi
            self.full_extend+=1
        else:
            deep_copy.reverse()
        

        self.lattice.extend(deep_copy)
    
    def copy_lattice(self):
        """
        Manually creates a deep copy of the lattice.
        
        This is necessary because standard copy.deepcopy() fails on the
        custom impactx pybind objects.
        
        Returns:
            list: A new list with new, independent copies of each element.
        """
        import impactx
        import copy

        copied_lattice = []
        for element in self.lattice:
            # Re-create the element based on its type to ensure it's a new object
            if isinstance(element, impactx.elements.ExactSbend):
                new_element = impactx.elements.ExactSbend(
                    ds=element.ds,
                    phi=np.rad2deg(element.phi), # impactx constructor takes degrees, but attribute is in rads
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactQuad):
                new_element = impactx.elements.ExactQuad(
                    ds=element.ds,
                    k=element.k,
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactDrift):
                new_element = impactx.elements.ExactDrift(
                    ds=element.ds,
                    nslice=element.nslice
                )
            else:
                # For other non-impactx types (like BeamMonitor), a shallow copy is usually safe
                try:
                    new_element = copy.copy(element)
                except TypeError:
                    print(f"Warning: Could not copy element of type {type(element)}")
                    continue
            
            # Ensure other attributes like the name are also copied
            if hasattr(element, 'name'):
                new_element.name = element.name

            copied_lattice.append(new_element)
            
        return copied_lattice
    
    def plot_evolution(self):

        from matplotlib import pyplot as plt
        
        evol = self.evolution
        
        # prepare plot
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]

        # plot energy
        axs[0,0].plot(evol.location, evol.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy [GeV]')
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_xlim(long_limits)
        
        # plot charge
        axs[0,1].plot(evol.location, abs(evol.charge[0]) * np.ones(evol.location.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(evol.location, abs(evol.charge) * 1e9, color=col1)
        axs[0,1].set_ylabel('Charge [nC]')
        axs[0,1].set_xlim(long_limits)
        axs[0,1].set_ylim(0, abs(evol.charge[0]) * 1.3 * 1e9)
        
        # plot normalized emittance
        axs[0,2].plot(evol.location, evol.emit_ny*1e6, color=col2)
        axs[0,2].plot(evol.location, evol.emit_nx*1e6, color=col1)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_xlim(long_limits)
        axs[0,2].set_yscale('log')
        
        # plot energy spread
        axs[1,0].plot(evol.location, evol.rel_energy_spread*1e2, color=col1)
        axs[1,0].set_ylabel('Energy spread, rms [%]')
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_xlim(long_limits)
        axs[1,0].set_yscale('log')

        # plot bunch length
        axs[1,1].plot(evol.location, evol.bunch_length*1e6, color=col1)
        axs[1,1].set_ylabel(r'Bunch length, rms [$\mathrm{\mu}$m]')
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_xlim(long_limits)

        # plot beta function
        axs[1,2].plot(evol.location, evol.beta_y, color=col2)  
        axs[1,2].plot(evol.location, evol.beta_x, color=col1)
        axs[1,2].set_ylabel('Beta function [m]')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)
        axs[1,2].set_yscale('log')
        
        # plot transverse offset
        axs[2,0].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,0].plot(evol.location, evol.x*1e6, color=col1)
        axs[2,0].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        
        # plot dispersion
        axs[2,1].plot(evol.location, evol.dispersion_y*1e3, color=col2)  
        axs[2,1].plot(evol.location, evol.dispersion_x*1e3, color=col1)
        #axs[2,1].plot(evol.location, evol.second_order_dispersion_x*1e3, ':', color=col1)
        axs[2,1].set_ylabel('First-order dispersion [mm]')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_xlim(long_limits)

        # plot beam size
        axs[2,2].plot(evol.location, evol.beam_size_y*1e6, color=col2)  
        axs[2,2].plot(evol.location, evol.beam_size_x*1e6, color=col1)
        axs[2,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)
        
        
        plt.show()
    
    def plot_simplified(self, return_end_values=False):
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56
        ls, inv_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True)

        beta_x, alpha_x, _ = evolve_beta_function(ls=ls, ks=ks, beta0=self.beta0_x, alpha0=self.alpha0_x, inv_rhos=inv_rs, plot=True)
        beta_y, alpha_y, _ = evolve_beta_function(ls=ls, ks=-ks, beta0=self.beta0_y, alpha0=self.alpha0_y, plot=True)
        print(self.Dx0, self.Dpx0)
        D, Dp, _ = evolve_dispersion(ls=ls, ks=ks, inv_rhos=inv_rs, Dx0=self.Dx0, Dpx0=self.Dpx0, plot=True)
        R56, _ = evolve_R56(ls=ls, ks=ks, inv_rhos=inv_rs, plot=True, R560=self.R560)
        if return_end_values:
            return beta_x, alpha_x, beta_y, alpha_y, D, Dp, R56