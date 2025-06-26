import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize

class DriverDelaySystem(BeamDeliverySystem):

    def __init__(self, E_nom=1, delay_per_stage=1, length_stage=10, num_stages=2, ks=[], \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, reduced_lattice=True,
                 focus_at_trough = "x", fill_factor=0.7):
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

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.fill_factor = fill_factor
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(self.delay_per_stage)
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        # Use lengths to get the required B-field
        self.B_dipole = self.get_B_field(E_nom/1e9)

        self.reduced_lattice = reduced_lattice

        if self.reduced_lattice:
            self.cell_length = 2*(self.l_quads + self.l_dipole*2 + self.l_straight)
        else:
            self.cell_length = self.l_quads + self.l_dipole*2 + self.l_straight

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
        if self.reduced_lattice:
            self.cell_length = 2*(self.l_quads + self.l_dipole*2 + self.l_straight)
        else:
            self.cell_length = self.l_quads + self.l_dipole*2 + self.l_straight
        self.B_dipole = self.get_B_field(self.E_nom/1e9)

    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(value/1e9)
        if self.reduced_lattice:
            self.cell_length = 2*(self.l_quads + self.l_dipole*2 + self.l_straight)
        else:
            #self.cell_length = self.l_quads + self.l_dipole*2 + self.l_straight
            self.cell_length = self.l_dipole + self.l_straight
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
    
    def get_lattice_half(self, ks: list=[]):
        """
        Oscillating Chicanes:
            2 dipoles, 2-4 quadrupoles of the crests and troughs. Start at the trough.
            Max dipole field, 1.2.
            Assume no drift between dipoles and quads, for now.

        Input:
            ks: list of focusing strengths
        Returns:
            list of lattice elements
        """

        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        r = p/self.B_dipole # B in Tesla
        bend_angle = self.l_dipole/r
        bend_angle = np.rad2deg(bend_angle)

        import impactx

        quads = []
        lattice = []

        ns = 50 # Number of slices
        # First quad
        first_quad = impactx.elements.ExactQuad(k=ks[0], ds = self.l_quads/2, nslice=ns)
        quads.append(first_quad)

        # Last (middle) quad
        last_quad = impactx.elements.ExactQuad(k=ks[-1], ds = self.l_quads/2, nslice=ns)

        # Quads
        if len(ks)>2:
            for k_ in ks[1:-1]:
                quads.append(impactx.elements.ExactQuad(k=k_, ds = self.l_quads, nslice=ns))
        # Append last quad
        quads.append(last_quad)

        # Bend
        bend_up = impactx.elements.ExactSbend(name="bend up", ds=self.l_dipole, phi=bend_angle, B=self.B_dipole, nslice=ns)
        bend_down = impactx.elements.ExactSbend(name="bend down", ds=self.l_dipole, phi=-bend_angle, B=-self.B_dipole, nslice=ns)

        # Drift
        drift = impactx.elements.ExactDrift(name="drift", ds=self.l_drift, nslice=ns)
        drift_reduced_lattice = impactx.elements.ExactDrift(name="drift", ds=self.l_straight, nslice=ns)
        drift_2quads = impactx.elements.ExactDrift(name="drift2", ds=self.l_straight-2*self.l_quads, nslice=ns)

        # Make lattice. No room between dipoles at the crests and troughs
        if self.reduced_lattice:
            #reduced lattice
            lattice.append(quads[0])
            lattice.append(bend_up)
            lattice.append(drift_reduced_lattice)
            lattice.append(bend_down)
            lattice.append(quads[1])
        else:
            if len(quads)==3:
            # For attempting more focusing in lattice
                lattice.append(quads[0])
                lattice.append(bend_up)
                lattice.append(drift)
                lattice.append(quads[1])
                lattice.append(drift)
                lattice.append(bend_down)
                lattice.append(quads[2])
            elif len(quads)==4:
            # Lattice with 2 quads during the straights
                lattice.append(quads[0])
                lattice.append(bend_up)
                lattice.append(quads[1])
                lattice.append(drift_2quads)
                lattice.append(quads[2])
                lattice.append(bend_down)
                lattice.append(quads[3])
            
        return lattice


    def get_lattice_stage(self, ks: list=[]):
        """
        Oscillating Chicanes:
            4 dipoles, 70% fill factor, quadrupoles of the crests and troughs. Start at the trough.
            Max dipole field, 1.2.
            Assume no drift between dipoles and quads, for now.
            2 stages corresponds to 1 full "oscillation"

        Input:
            kx: focusing strength in x
            ky: focusing strength in y
            B_bend: dipole field [Tesla]
        Returns:
            list of lattice elements
        """

        lattice = []
        first_half = self.get_lattice_half(ks)
        lattice.extend(first_half)
        lattice.extend(first_half.reverse())
            
        return lattice
    
    def get_lattice(self, ks: list):
        lattice = self.get_lattice_stage(ks)*(self.num_stages//2)
        return lattice
    
    def match_quads(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        # Initial guess
        ff = self.cell_length/2
        if self.initial_focus_plane=="x":
            ff=ff
        else:
            ff=-ff
        #if beta0_y > beta0_x:
        fd = -ff
        kx0 = 1/ff/self.l_quads
        ky0 = 1/fd/self.l_quads


        def minimize_periodic(params):
            lattice = self.get_lattice_half(params)
            """
            Match dispersion and R56.
            params = [k1, k2, ...]
            """
            ls = []
            ks = []
            inverse_rs = []

            for element in lattice:
                ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)
                if hasattr(element, "phi"):
                    phi = element.phi # Given in rad for some reason (input takes degrees)
                    inverse_r = phi/self.l_dipole
                    inverse_rs.append(inverse_r)
                else:
                    inverse_rs.append(0)
            _, Dp, _ = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            R56, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            # D´ is unitless, R56 is m/E. Normalise with energy per length if stage?
            R56_norm = self.E_nom*SI.e / self.cell_length
            return Dp**2 + (R56*R56_norm)**2
        
        if self.reduced_lattice:
            res = minimize(minimize_periodic, x0=[kx0, ky0])
        else:
            res = minimize(minimize_periodic, x0=[kx0, ky0, kx0, ky0])
            #res = minimize(minimize_periodic, x0=[kx0, ky0, kx0])
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
            lattice = self.get_lattice_half(self.ks)

            ls = []
            ks = []

            for element in lattice:
                ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)

            beta_x, alpha_x, _ = evolve_beta_function(ls, np.array(ks), beta0=params[0], alpha0=0, fast=True)
            beta_y, alpha_y, _ = evolve_beta_function(ls, -np.array(ks), beta0=params[1], alpha0=0, fast=True)

            return (beta_x-params[0])**2/self.cell_length**2 + (alpha_x)**2 + \
                (beta_y-params[1])/self.cell_length**2 + (alpha_y)**2
        
        res = minimize(minimize_periodic, x0=beta0)

        print(res)
        betas = res.x
        return betas
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx
        from abel.utilities.beam_physics import evolve_beta_function

        # Get lattice
        if not list(self.ks):
            self.ks = self.match_quads()
        
        self.lattice = self.get_lattice(self.ks)
        """
        Evolve beta function here
        """
        ls=[]
        ks=[]
        for element in self.lattice:
                ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)
        _, _, evo = evolve_beta_function(ls, ks, beta0=beam0.beta_x(), alpha0=beam0.alpha_x(), plot=True)

        # run ImpactX
        beam, evol = run_impactx(self.lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data,\
                                  space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        if self.reduced_lattice:
            return self.cell_length*(self.num_stages//2)
        else:
            return self.cell_length*self.num_stages
    
    def get_nom_energy(self):
        return self.E_nom
    
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











