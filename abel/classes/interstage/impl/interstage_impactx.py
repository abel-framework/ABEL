from abel.classes.interstage.interstage import Interstage
import numpy as np
import os
from types import SimpleNamespace
import scipy.constants as SI

class InterstageImpactX(Interstage):
    
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None, beta0=None, R56=0):
        
        super().__init__(nom_energy=nom_energy, dipole_length=dipole_length, dipole_field=dipole_field, beta0=beta0)
        
        self.R56 = R56

        self.enable_nonlinearity = True
        self.enable_sextupole = True
        self.enable_chicane = False
        
        self.enable_space_charge = False
        self.enable_csr = False
        self.enable_isr = False
        
        self.L2_by_L1 = 1
        self.B2_by_B1 = np.sqrt(2)
        
        self.keep_data = False

        self.evolution = SimpleNamespace()

    
     # lattice length
    def get_length(self):
        return self.dipole_length*(2+2*self.L2_by_L1)

    def get_lattice(self):

        import impactx
        
        # initialize lattice
        lattice = []
        
        if not self.dipole_length:
            L1 = 0
        else:
            L1 = self.dipole_length
        
        if not self.dipole_field:
            B1 = 0
        else:
            B1 = self.dipole_field

        L2 = self.L2_by_L1*L1

        E0 = self.nom_energy
        gamma0 = E0*SI.e/(SI.m_e*SI.c**2);
        v0 = SI.c*np.sqrt(1-1/gamma0**2);
        p0 = SI.m_e*gamma0*v0;

        # define dipole
        Dx = L1**2*B1*SI.e/(2*p0)
        phi = B1*L1*SI.e/p0
        ns = 25  # number of slices per ds in the element
        bend = impactx.elements.ExactSbend(name="dipole", ds=L1, phi=np.rad2deg(phi), B=B1, nslice=ns)
        
        # define plasma lens
        if L1 or L2:
            f0 = L1*L2/(L1+L2)
            k=1/f0
        else:
            f0 = 0
            k = np.inf
        dtaper = int(self.enable_nonlinearity)*1/Dx
        pl = impactx.elements.TaperedPL(k=k, taper=dtaper, name="plasmalens")

        B2 = int(self.enable_chicane)*self.B2_by_B1*B1
        B2_div_B1 = int(self.enable_chicane)*self.B2_by_B1
        if abs(B2) > 0:
            phi2 = B2*(L2/2)*(SI.e/p0)
            chicane1 = impactx.elements.ExactSbend(name="chicane1", ds=L2/2, phi=np.rad2deg(phi2), B=B2, nslice=ns)
        else:
            chicane1 = impactx.elements.ExactDrift(name="chicane1", ds=L2/2, nslice=ns)

        B3 = -B2 - B1*(1/self.L2_by_L1)*(1-1/self.L2_by_L1)
        if abs(B3) > 0:
            phi3 = B3*(L2/2)*(SI.e/p0)
            chicane2 = impactx.elements.ExactSbend(name="chicane2", ds=L2/2, phi=np.rad2deg(phi3), B=B3, nslice=ns)
        else:
            chicane2 = impactx.elements.ExactDrift(name="chicane2", ds=L2/2, nslice=ns)
    
        Dx_mid = Dx*(1/4)*(1 + 3*self.L2_by_L1 + 2*(B2_div_B1)*self.L2_by_L1**2)
        DDxp_mid = (2*B1*L1*SI.e/p0)*((3/4)*(1/self.L2_by_L1 + 1) - 1)*(2.0-int(self.enable_nonlinearity))
        m_sext = int(self.enable_sextupole)*2*DDxp_mid/Dx_mid**2
        sextupole = impactx.elements.Multipole(multipole=3, K_normal=m_sext, K_skew=0, name="sextupole")
        
        # add beam diagnostics
        #monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

        # specify the lattice sequence
        #lattice.append(monitor)
        lattice.append(bend)
        #lattice.append(monitor)
        lattice.append(pl)
        #lattice.append(monitor)
        lattice.extend([chicane1, chicane2]) #lattice.extend([chicane1, monitor, chicane2])
        #lattice.append(monitor)
        lattice.append(sextupole)
        #lattice.append(monitor)
        lattice.extend([chicane2, chicane1]) #lattice.extend([chicane2, monitor, chicane1])
        #lattice.append(monitor)
        lattice.append(pl)
        #lattice.append(monitor)
        lattice.append(bend)
        #lattice.append(monitor)
        
        return lattice

        
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
        axs[2,1].plot(evol.location, evol.second_order_dispersion_x*1e3, ':', color=col1)
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
        