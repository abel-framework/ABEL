import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize

class DriverDelaySystem(BeamDeliverySystem):

    def __init__(self, E_nom=1, delay_per_stage=1, length_stage=10, num_stages=2, ks=[], \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, \
                 focus_at_trough = "x", fill_factor=0.7, mirrored=True, layoutnr = 1, use_monitors=False):
        # E_nom in eV, delay in ns
        super().__init__()
        self._E_nom = E_nom #backing variable 
        self._delay_per_stage = delay_per_stage/1e9
        self.num_stages = num_stages
        self._length_stage = length_stage
        self.keep_data = keep_data
        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.ks = ks
        self.mirrored = mirrored
        self.layoutnr = layoutnr
        self.use_monitors = use_monitors

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.fill_factor = fill_factor
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(self.delay_per_stage)
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        # Use lengths to get the required B-field
        self.B_dipole = self.get_B_field(E_nom/1e9)

        self.cell_length = self.get_length()/4

        self.initial_focus_plane = focus_at_trough

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
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(self.delay_per_stage/1e9)
        self.cell_length = self.get_length()/4
        self.B_dipole = self.get_B_field(self.E_nom/1e9)

    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(value/1e9)
        self.cell_length = self.get_length()/4
        self.B_dipole=self.get_B_field(self.E_nom/1e9)


    # Setting the new E_nom
    @E_nom.setter
    def E_nom(self, value):
        self._E_nom = value
        # Recalculate B if new energy is set
        print("Recalculating B-field")
        self.B_dipole=self.get_B_field(value/1e9)
        print("New B-field: ", self.B_dipole)

    def get_B_field(self, p):
        # p in GeV, B given in Tesla
        func = lambda B: self.l_quads + self.l_straight*np.cos(self.l_dipole*0.3*B/p)\
              + 2*p/0.3/B*np.sin(self.l_dipole*0.3*B/p) - self.length_stage
        
        from scipy.optimize import root_scalar
        print(root_scalar(func, x0=0.5))
        return root_scalar(func, x0=0.5).root

    def get_lattice_lengths(self, delay):
        l_dipole = self.fill_factor/2*(SI.c*delay + self.length_stage)
        l_straight = SI.c*delay + self.length_stage - self.l_quads - 2*l_dipole
        l_drift = 1/2*(l_straight - self.l_quads)
        return l_dipole, l_straight, l_drift
    
    
    def list2lattice(self, ls: list, ks: list, phis: list):
        import impactx
        lattice=[]
        if self.use_monitors:
            lattice.append(impactx.elements.BeamMonitor(name="monitor", backend="h5", encoding="g"))
        ns = 50
        for i, l in enumerate(ls):
            if phis[i]!=0:
                element = impactx.elements.ExactSbend(ds=l, phi=np.rad2deg(phis[i]), nslice=ns)
            elif ks[i]!=0:
                element = impactx.elements.ExactQuad(ds=l, k=ks[i], nslice=ns)
            else:
                element = impactx.elements.ExactDrift(ds=l, nslice=ns)
            
            lattice.append(element)

            if self.use_monitors:
                lattice.append(impactx.elements.BeamMonitor(name="monitor", backend="h5", encoding="g"))

        return lattice

    def inv_r2phi(self, inv_rs: list):
        phis=[]
        for inv_r in inv_rs:
            phi = self.l_dipole*inv_r
            phis.append(phi)
        return phis
    
    def phi2inv_r(self, phis: list):
        inverse_rs = []
        for phi in phis:
            inv_r = phi/self.l_dipole
            inverse_rs.append(inv_r)
        return inverse_rs

    
    def get_lattice2stages(self, ks, return_lists=False):
        """
        Make list of ls, ks, and phis (basically predetermined), and list2lattice, to get final lattice
        """
        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        r = p/self.B_dipole # B in Tesla
        bend_angle = self.l_dipole/r
        bend_angle = np.rad2deg(bend_angle)

        match self.layoutnr:
            case 1:
                # Set lengths
                ls = [self.l_quads/2, self.l_dipole, self.l_quads, self.l_straight-2*self.l_quads, self.l_quads, self.l_dipole, self.l_quads/2]
                ls2 = ls.copy()
                ls2.reverse()
                ls.extend(ls2)

                # Set the ks in a list
                ks = [ks[0], 0, ks[1], 0, ks[2], 0, ks[3], ks[3], 0, ks[4], 0, ks[5], 0, ks[0]]

                # Set phis
                phis = [0, bend_angle, 0, 0, 0, -bend_angle, 0]
                phis2 = phis.copy()
                phis2.reverse()
                phis.extend(phis2)

            case 2:
                drift = (self.l_straight-3*self.l_quads)/2
                ls = [self.l_quads/2, self.l_dipole, self.l_quads, drift, self.l_quads, drift, self.l_quads, self.l_dipole, self.l_quads/2]
                ls2 = ls.copy()
                ls2.reverse()
                ls.extend(ls2)

                # Set the ks in a list (1 extra quad in the straight section)
                ks = [ks[0], 0, ks[1], 0, ks[2], 0, ks[3], 0, ks[4], ks[4], 0, ks[4], 0, ks[5], 0, ks[6],0, ks[0]]

                # Set phis
                phis = [0, bend_angle, 0, 0, 0, 0, 0, -bend_angle, 0]
                phis2 = phis.copy()
                phis2.reverse()
                phis.extend(phis2)
            case _:
                raise ValueError("Invalid value for attribute: layoutnr")
        
        if return_lists:
            return self.list2lattice(ls=ls, ks=ks, phis=phis), ls, ks, phis
        else:
            return self.list2lattice(ls=ls, ks=ks, phis=phis)
        
    def get_lattice(self, ks):
        return self.get_lattice2stages(ks)*(self.num_stages//2)
    
    def match_quads(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        # Initial guess
        ff = self.cell_length*2/4
        if self.initial_focus_plane=="x":
            ff=ff
        else:
            ff=-ff
        #if beta0_y > beta0_x:
        fd = -ff
        kx0 = 1/ff/self.l_quads
        ky0 = 1/fd/self.l_quads

        def minimize_periodic(params):
            _, ls, ks, phis = self.get_lattice2stages(params, return_lists=True)

            inverse_rs = self.phi2inv_r(phis)
            """
            Match dispersion and R56.
            params = [k1, k2, ...]
            """

            D, Dp, _ = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            R56, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            # D´ is unitless, R56 is m/E. Normalise with energy per length
            R56_norm = self.E_nom/1e9 / self.cell_length
            #print(D**2, "  ", (R56*R56_norm)**2)
            return (R56*R56_norm)**2 + (D/self.cell_length)**2
        
        match self.layoutnr:
            case 1:
                res = minimize(minimize_periodic, x0=[kx0, ky0]*3)
            case 2:
                res = minimize(minimize_periodic, x0=[kx0, ky0]*4)
            case _:
                raise ValueError("Invalid value for attribute: layoutnr")
        print(res)
        self.ks = res.x
    
    def match_betas(self):
        """
        Use the quad-values found earlier, to find periodic beta/alpha-function
        """
        if not list(self.ks):
            self.match_quads()

        from abel.utilities.beam_physics import evolve_beta_function
        # Initial guess
        beta0 = [5, 5]

        def minimize_periodic(params):
            """
            params = [beta_x, beta_y]
            """
            _, ls, ks, _ = self.get_lattice2stages(self.ks, return_lists=True)                

            beta_x, _, _ = evolve_beta_function(ls, np.array(ks), beta0=params[0], alpha0=0, fast=True)
            beta_y, _, _ = evolve_beta_function(ls, -np.array(ks), beta0=params[1], alpha0=0, fast=True)

            return (beta_x-params[0])**2/self.cell_length**2 + (beta_y-params[1])**2/self.cell_length**2 
        
        res = minimize(minimize_periodic, x0=beta0, bounds=[(1e-3, 200)]*2)

        print(res)
        betas = res.x
        return betas
    
    def match_quads_and_betas(self):
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56
        
        init = [5, 5] # betas

        ff = self.cell_length*2/4
        if self.initial_focus_plane=="x":
            ff=ff
        else:
            ff=-ff
        #if beta0_y > beta0_x:
        fd = -ff
        kx0 = 1/ff/self.l_quads
        ky0 = 1/fd/self.l_quads

        match self.layoutnr:
            case 1:
                init.extend([kx0, ky0]*3)
            case 2:
                init.extend([kx0, ky0]*4)

        def minimize_periodic(params):
            beta_x0 = params[0]
            beta_y0 = params[1]

            ks = params[2:]

            _, ls, ks, phis = self.get_lattice2stages(ks=ks, return_lists=True)

            inverse_rs = self.phi2inv_r(phis)

            D, _, _ = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)
            R56, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            # D´ is unitless, R56 is m/E. Normalise with energy per length
            R56_norm = self.E_nom/1e9 / self.cell_length

            beta_x, _, _ = evolve_beta_function(ls, np.array(ks), beta0=beta_x0, alpha0=0, fast=True)
            beta_y, _, _ = evolve_beta_function(ls, -np.array(ks), beta0=beta_y0, alpha0=0, fast=True)

            return (R56*R56_norm)**2 + (D/self.cell_length)**2 + \
                (beta_x-beta_x0)**2/self.cell_length**2 + (beta_y-beta_y0)**2/self.cell_length**2
        res = minimize(minimize_periodic, x0 = init)
        print(res)
        self.ks = res.x[2:]
        return res.x[0], res.x[1]

    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56

        # Get lattice
        if not list(self.ks):
            self.ks = self.match_quads()
        
        self.lattice = self.get_lattice(self.ks)
        print(self.lattice)
        """
        Evolve beta function here
        """
        
        ls, ks, inverse_rs = self.get_element_attributes(self.lattice)

        _, _, evox = evolve_beta_function(ls, ks, beta0=beam0.beta_x(), alpha0=beam0.alpha_x(), plot=True)
        _, _, evoy = evolve_beta_function(ls, -np.array(ks), beta0=beam0.beta_y(), alpha0=beam0.alpha_y(), plot=True)
        _, _, evoD = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, plot=True,)
        #_, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, plot=True)

        # run ImpactX
        beam, evol = run_impactx(self.lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data,\
                                  space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return 2*(self.l_quads + 2*self.l_dipole + self.l_straight)
    
    def get_nom_energy(self):
        return self.E_nom
    
    def get_element_attributes(self, lattice):
        ls = []
        ks = []
        inverse_rs = []
        for element in lattice:
                if hasattr(element, "l"):
                    ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)
                if hasattr(element, "phi"):
                    phi = element.phi # Given in rad for some reason (although input takes degrees)
                    inverse_r = phi/self.l_dipole
                    inverse_rs.append(inverse_r)
                else:
                    inverse_rs.append(0)
        return ls, ks, inverse_rs
    
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











