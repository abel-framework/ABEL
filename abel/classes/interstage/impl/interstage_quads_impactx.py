from abel.classes.interstage.interstage import Interstage
import numpy as np
import contextlib, os
from types import SimpleNamespace

import scipy.constants as SI

class InterstageQuadsImpactX(Interstage):
    
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None, beta0=None):
        
        super().__init__(nom_energy=nom_energy, dipole_length=dipole_length, dipole_field=dipole_field, beta0=beta0)
        
        self.enable_space_charge = False
        self.enable_csr = False
        self.enable_isr = False
        
        self.keep_data = False
        
        self._quad_strength1 = None
        self._quad_strength2 = None
        self.B2_by_B1 = -0.89785923
        self.L2_by_L1 = 0.25

        self.evolution = SimpleNamespace()

    @property
    def quad_strength1(self) -> float:
        if self._quad_strength1 is None:
            return -4.34019779/self.dipole_length**2
        else:
            return self._quad_strength1
    @quad_strength1.setter
    def quad_strength1(self, val):
        self._quad_strength1 = val

    @property
    def quad_strength2(self) -> float:
        if self._quad_strength2 is None:
            return 5.77312688/self.dipole_length**2
        else:
            return self._quad_strength2
    @quad_strength2.setter
    def quad_strength2(self, val):
        self._quad_strength2 = val
        
    
     # lattice length
    def get_length(self):
        return self.dipole_length*(2+2.4)

    
    def get_lattice(self, kq1=None, kq2=None, B2=None, half_lattice=False, exact=True):

        import impactx
        
        # initialize lattice
        lattice = []
        
        L1 = self.dipole_length
        B1 = self.dipole_field
        L2 = L1*self.L2_by_L1

        if kq1 is None:
            kq1 = self.quad_strength1
        if kq2 is None:
            kq2 = self.quad_strength2
        if B2 is None:
            B2 = self.dipole_field*self.B2_by_B1
            
        E0 = self.nom_energy
        gamma0 = E0*SI.e/(SI.m_e*SI.c**2);
        v0 = SI.c*np.sqrt(1-1/gamma0**2);
        p0 = SI.m_e*gamma0*v0;

        # number of slices per ds in the element
        ns = 25
        
        # define dipole
        dipole_radcurv1 = p0/(B1*SI.e)
        if exact:
            bend1 = impactx.elements.ExactSbend(name="dipole1", ds=L1, B=B1, phi=np.rad2deg(L1/dipole_radcurv1), nslice=ns)
        else:
            bend1 = impactx.elements.Sbend(name="dipole1", ds=L1, rc=dipole_radcurv1, nslice=ns)

        # define first quad
        Lq = L1*0.4;
        if exact:
            quad1 = impactx.elements.ExactQuad(name="quad1", ds=Lq, k=kq1, nslice=ns)
        else:
            quad1 = impactx.elements.Quad(name="quad1", ds=Lq, k=kq1, nslice=ns)

        # define drift
        dipole_radcurv2 = p0/(B2*SI.e)
        if exact:
            bend2 = impactx.elements.ExactSbend(name="dipole2", ds=L2, B=B2, phi=np.rad2deg(L2/dipole_radcurv2), nslice=ns)
        else:
            bend2 = impactx.elements.Sbend(name="dipole2", ds=L2, rc=dipole_radcurv2, nslice=ns)
        
        # define second quad
        if exact:
            half_quad2 = impactx.elements.ExactQuad(name="quad2", ds=Lq/2, k=kq2, nslice=ns)
        else:
            half_quad2 = impactx.elements.Quad(name="quad2", ds=Lq/2, k=kq2, nslice=ns)

        if exact:
            drift = impactx.elements.ExactDrift(name="drift", ds=L1*0.001, nslice=2)
        else:
            drift = impactx.elements.Drift(name="drift", ds=L1*0.001, nslice=2)
        
        # specify the lattice sequence
        lattice.append(bend1)
        lattice.append(quad1)
        lattice.append(bend2)
        lattice.append(half_quad2)
        lattice.append(drift)
        if not half_lattice:
            lattice.append(drift)
            lattice.append(half_quad2)
            lattice.append(bend2)
            lattice.append(quad1) 
            lattice.append(bend1)
        
        return lattice


    def match_lattice(self, runnable=None):

        from scipy import optimize as optim
        from abel.apis.impactx.impactx_api import run_envelope_impactx
        
        # ensure that the incoming beta function is set before matching
        assert self.beta0 is not None
        
        from impactx import twiss, distribution
        
        # define the matching merit function
        def match_fcn(params):
            lattice = self.get_lattice(kq1=params[0], kq2=params[1], B2=params[2], half_lattice=True, exact=False)
    
            # make particle distribution
            distr = distribution.KVdist(
                **twiss(
                    beta_x=self.beta0, alpha_x=0.0, emitt_x=1.0e-6,
                    beta_y=self.beta0, alpha_y=0.0, emitt_y=1.0e-6,
                    beta_t=0.5, alpha_t=0.0, emitt_t=1.0e-12,
                )
            )
            
            # envelope tracking using ImpactX
            evol = run_envelope_impactx(lattice, distr, nom_energy=self.nom_energy, runnable=runnable, verbose=False)

            # extract Twiss parameters and first order dispersions
            last_index = len(evol.location)
            ds = evol.location[last_index-1]-evol.location[last_index-2]
            beta_estimate = self.dipole_length**2/self.beta0
            dispersion_estimate = self.dipole_length**2*self.dipole_field*SI.c/(self.nom_energy)
            alpha_x = -0.5*(evol.beta_x[last_index-1]-evol.beta_x[last_index-2])/ds / beta_estimate
            alpha_y = -0.5*(evol.beta_y[last_index-1]-evol.beta_y[last_index-2])/ds / beta_estimate
            dispersion_prime_x = (evol.dispersion_x[last_index-1]-evol.dispersion_x[last_index-2])/ds / dispersion_estimate
            
            merit = alpha_x**2 + alpha_y**2 + dispersion_prime_x**2
            return merit

        # perform fit
        guess = [self.quad_strength1, self.quad_strength2, self.dipole_field*self.B2_by_B1]
        fit_params = optim.minimize(match_fcn, guess, options={'eps': 1e-4})
        
        # set the match
        self.quad_strength1 = fit_params.x[0]
        self.quad_strength2 = fit_params.x[1]
        self.B2_by_B1 = fit_params.x[2]/self.dipole_field
        
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx
        
        # get lattice
        lattice = self.get_lattice()
        
        # run ImpactX
        beam, evol = run_impactx(lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def plot_layout(self):

        from matplotlib import pyplot as plt
        
        L1 = float(self.dipole_length)
        B1 = float(self.dipole_field)
        L2 = self.L2_by_L1*L1
        B2 = float(int(self.enable_chicane)*self.B2_by_B1*B1)
        B3 = -B2 - B1*(L1/L2)*(1-L1/L2)
        
        nparts = 25;
        
        rho1 = self.nom_energy/(SI.c*B1)
        
        x0 = 0
        y0 = 0
        theta0 = 0*L1/rho1

        
        ss1 = np.linspace(0,L1,nparts);
        thetas1 = ss1/abs(rho1)
        xs_bend1 = rho1*np.sin(thetas1)
        ys_bend1 = rho1*(1-np.cos(thetas1))
        xs1 = x0 + xs_bend1*np.cos(theta0) + ys_bend1*np.sin(theta0)
        ys1 = y0 - xs_bend1*np.sin(theta0) + ys_bend1*np.cos(theta0)
        xs = xs1
        ys = ys1
        theta1 = theta0-thetas1[-1]
        
        rho2 = self.nom_energy/(SI.c*B2)
        ss2 = np.linspace(0,L2/2,nparts);
        thetas2 = ss2/abs(rho2)
        xs_bend2 = rho2*np.sin(thetas2)
        ys_bend2 = rho2*(1-np.cos(thetas2))
        xs2 = xs1[-1] + xs_bend2*np.cos(theta1) + ys_bend2*np.sin(theta1)
        ys2 = ys1[-1] - xs_bend2*np.sin(theta1) + ys_bend2*np.cos(theta1)
        xs = np.append(xs,xs2)
        ys = np.append(ys,ys2)
        theta2 = theta1-thetas2[-1]

        rho3 = self.nom_energy/(SI.c*B3)
        ss3 = np.linspace(0,L2/2,nparts);
        thetas3 = ss3/abs(rho3)
        xs_bend3 = abs(rho3)*np.sin(thetas3)
        ys_bend3 = (rho3)*(1-np.cos(thetas3))
        xs3 = xs2[-1] + xs_bend3*np.cos(theta2) + ys_bend3*np.sin(theta2)
        ys3 = ys2[-1] - xs_bend3*np.sin(theta2) + ys_bend3*np.cos(theta2)
        xs = np.append(xs,xs3)
        ys = np.append(ys,ys3)
        theta3 = theta0+theta2-thetas3[-1]
        
        xs_rest = -(np.flip(xs) - xs[-1])
        ys_rest = (np.flip(ys) - ys[-1])
        xs4 = xs3[-1] + xs_rest*np.cos(theta3) + ys_rest*np.sin(theta3)
        ys4 = ys3[-1] - xs_rest*np.sin(theta3) + ys_rest*np.cos(theta3)
        xs = np.append(xs,xs4)
        ys = np.append(ys,ys4)
        
        # prepare plot
        fig, ax = plt.subplots(1)
        fig.set_figwidth(14)
        fig.set_figheight(6)

        # plot nominal orbit
        ax.plot(xs, ys, 'g')
        #ax.plot([x0, xs1[-1], xs2[-1], xs3[-1]], [y0, ys1[-1], ys2[-1], ys3[-1]], 'ko')
        ax.axis('equal')
        
    
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
        
        # plot transverse offset
        axs[2,0].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,0].plot(evol.location, evol.y*1e6, color=col1)
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
        