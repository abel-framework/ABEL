import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize

class DriverDelaySystem(BeamDeliverySystem):

    def __init__(self, E_nom=1, delay_per_stage=1, length_stage=10, num_stages=2, kx=None, ky=None, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, reduced_lattice=True,
                 focus_at_trough = "x"):
        # E_nom in eV, delay in ns
        super().__init__()
        self._E_nom = E_nom #backing variable 
        self._delay_per_stage = delay_per_stage/1e9
        self.num_stages = num_stages
        self.length_stage = length_stage
        self.kx = kx
        self.ky = ky
        self.keep_data = keep_data
        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.fill_factor = 0.7
        self.l_dipole = self.fill_factor/2*(SI.c*self.delay_per_stage + self.length_stage)
        self.l_straight = SI.c*self.delay_per_stage + self.length_stage - self.l_quads - 2*self.l_dipole
        self.l_drift = 1/2*(self.l_straight - self.l_quads)
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        # Use lengths to get the required B-field
        self.B_dipole = self.get_B_field(E_nom/1e9)

        self.reduced_lattice = reduced_lattice

        if self.reduced_lattice:
            self.cell_length = 2*(self.l_quads*2 + self.l_dipole*2 + self.l_straight)
        else:
            self.cell_length = self.l_quads*2 + self.l_dipole*2 + self.l_straight

        self.focus_at_trough = focus_at_trough

    @property
    def E_nom(self):
        return self._E_nom
    
    @property
    def delay_per_stage(self):
        return self._delay_per_stage
    
    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value/1e9
        self.l_dipole = self.fill_factor/2*(SI.c*value/1e9 + self.length_stage)
        self.l_straight = SI.c*value/1e9 + self.length_stage - self.l_quads - 2*self.l_dipole
        self.l_drift = 1/2*(self.l_straight - self.l_quads)
        if self.reduced_lattice:
            self.cell_length = 2*(self.l_quads*2 + self.l_dipole*2 + self.l_straight)
        else:
            self.cell_length = self.l_quads*2 + self.l_dipole*2 + self.l_straight
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



    def get_lattice_single_stage(self, kx1, ky1, kx2=None, ky2=None):
        """
        Oscillating Chicanes:
            4 dipoles, 70% fill factor, quadrupoles of the crests and troughs. Start at the trough.
            Max dipole field, 1.2.
            Assume no drift between dipoles and quads, for now.

        Input:
            kx: focusing strength in x
            ky: focusing strength in y
            B_bend: dipole field [Tesla]
        Returns:
            list of lattice elements
        """

        p = self.E_nom /1e9 # eV to GeV
        r = p/0.3*self.B_dipole # p in GeV, B in Tesla
        bend_angle = self.l_dipole/r

        import impactx

        lattice = []

        ns = 25 # Number of slices

        # Quads
        quad_x_half = impactx.elements.ExactQuad(name="quad_x1", k=kx1/2, ds = self.l_quads/2, nslice=ns)
        quad_y1 = impactx.elements.ExactQuad(name="quad_y1", k=ky1, ds = self.l_quads, nslice=ns)
        if kx2 is not None and ky2 is not None:
            quad_x2 = impactx.elements.ExactQuad(name="quad_x2", k=kx2, ds = self.l_quads, nslice=ns)
            quad_y2 = impactx.elements.ExactQuad(name="quad_y2", k=ky2, ds = self.l_quads, nslice=ns)

        # Bend
        bend_up = impactx.elements.ExactSbend(name="bend up", ds=self.l_dipole, phi=-bend_angle)
        bend_down = impactx.elements.ExactSbend(name="bend down", ds=self.l_dipole, phi=+bend_angle)

        # Drift
        drift = impactx.elements.ExactDrift(name="drift", ds=self.l_drift, nslice=ns)
        drift_reduced_lattice = impactx.elements.ExactDrift(name="drift", ds=self.l_straight, nslice=ns)

        # Make lattice. No room between dipoles at the crests and troughs
        if self.reduced_lattice:
            #reduced lattice
            lattice.append(quad_x_half)
            lattice.append(bend_up)
            lattice.append(drift_reduced_lattice)
            lattice.append(bend_down)
            lattice.append(quad_y1)
            lattice.append(bend_down)
            lattice.append(drift_reduced_lattice)
            lattice.append(bend_up)
            lattice.append(quad_x_half)
        else:
            # For attempting more focusing in lattice, seems unusable per now
            lattice.append(quad_x_half)
            lattice.append(bend_up)
            lattice.append(drift)
            lattice.append(quad_y1)
            lattice.append(drift)
            lattice.append(bend_down)
            lattice.append(quad_x2)
            lattice.append(bend_down)
            lattice.append(drift)
            lattice.append(quad_y2)
            lattice.append(drift)
            lattice.append(bend_up)
            lattice.append(quad_x_half)
        
        return lattice
    
    def get_lattice(self, kx1, ky1, kx2=None, ky2=None):
        if self.reduced_lattice:
            lattice = self.get_lattice_single_stage(kx1, ky1)*(self.num_stages//2)
        else:
            lattice = self.get_lattice_single_stage(kx1, ky1, kx2, ky2)*(self.num_stages//2)
        return lattice
    
    def match_quads(self, beam):
        from abel.utilities.beam_physics import evolve_beta_function
        beta0_x = beam.beta_x()
        alpha0_x = beam.alpha_x()
        beta0_y = beam.beta_y()
        alpha0_y = beam.alpha_y()
        # Initial guess
        ff = self.cell_length/2
        if self.focus_at_trough=="x":
            ff=ff
        else:
            ff=-ff
        #if beta0_y > beta0_x:
        #    ff = -ff
        fd = -ff
        print(r"$ff_{guess}$: ", ff)
        kx0 = 1/ff/self.l_quads
        ky0 = 1/fd/self.l_quads

        def minimize_periodic(params):
            kx1 = params[0]
            ky1 = params[1]
            if self.reduced_lattice:
                lattice = self.get_lattice_single_stage(kx1, ky1)
            else:
                kx2 = params[2]
                ky2 = params[3]
                lattice = self.get_lattice_single_stage(kx1, ky1, kx2, ky2)

            ls = []
            ks = []
            for element in lattice:
                ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)
            beta_x, alpha_x, _ = evolve_beta_function(ls, np.array(ks), beta0=beta0_x, alpha0=alpha0_x, fast=True)
            beta_y, alpha_y, _ = evolve_beta_function(ls, -np.array(ks), beta0=beta0_y, alpha0=alpha0_y, fast=True)
            return (beta_x-beta0_x)**2/self.cell_length**2 + (alpha_x-alpha0_x)**2 + \
                (beta_y-beta0_y)/self.cell_length**2 + (alpha_y-alpha0_y)**2
        if self.reduced_lattice:
            res = minimize(minimize_periodic, x0=[kx0, ky0])
        else:
            res = minimize(minimize_periodic, x0=[kx0, ky0, kx0, ky0])
        print(res)
        ks = res.x
        return ks
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx

        # Get lattice
        if self.kx is None or self.ky is None:
            ks = self.match_quads(beam0)
            self.kx1 = ks[0]
            self.ky1 = ks[1]
            if not self.reduced_lattice:
                self.kx2 = ks[2]
                self.ky2 = ks[3]

                lattice = self.get_lattice(self.kx1, self.ky1, self.kx2, self.ky2)
            else:
                lattice = self.get_lattice(self.kx1, self.ky1)
        
        # run ImpactX
        beam, evol = run_impactx(lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
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











