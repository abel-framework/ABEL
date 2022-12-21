from opal import CONFIG, Interstage
from opal.apis.elegant.elegant_api import elegant_run, elegant_apl_fieldmap2D
from opal.utilities.beamphysics import evolveBetaFunction, evolveDispersion, evolveSecondOrderDispersion
from opal.utilities import SI
from string import Template
from scipy.optimize import minimize
import tempfile
import numpy as np
import matplotlib.pyplot as plt

class InterstageELEGANT(Interstage):
    
    def __init__(self, E0 = None, beta0 = None, Ldip = None, Bdip = 1):
        self.E0 = E0
        self.beta0 = beta0
        self.Ldip = Ldip
        self.Bdip = Bdip
        self.g_max = 1000 # [T/m]
    
    # evaluate beta function 
    def initialBetaFunction(self):
        if callable(self.beta0):
            return self.beta0(self.E0)
        else:
            return self.beta0
        
    # evaluate dipole field
    def dipoleField(self):
        if callable(self.Bdip):
            return self.Bdip(self.E0)
        else:
            return self.Bdip
    
    # evaluate dipole length  
    def dipoleLength(self):
        if callable(self.Ldip):
            return self.Ldip(self.E0)
        else:
            return self.Ldip
    
    # spacer length
    def spacerLength(self):
        return 0.05 * self.dipoleLength()
    
    # plasma-lens length
    def lensLength(self):
        f = (self.dipoleLength()+2*self.spacerLength())/2
        k = self.g_max*SI.c/self.E0
        return 1/(k*f)
    
    # sextupole length
    def correctorLength(self):
        return 0.1 * self.dipoleLength()
    
    # sextupole length
    def sextupoleLength(self):
        return 0.2 * self.dipoleLength()
    
    # drift length
    def driftLength(self):
        return self.dipoleLength() + self.spacerLength() - self.correctorLength()
       
    # full lattice 
    def fullLattice(self, g_lens=0, B_corr=0, m_sext=0, tau_lens=0):
        
        # element length array
        dL = self.spacerLength()
        L_dip = self.dipoleLength()
        L_lens = self.lensLength()
        L_corr = self.correctorLength()
        L_drift = self.driftLength()
        L_sext = self.sextupoleLength()
        ls = np.array([dL, L_dip, dL, L_lens, dL, L_corr, L_drift, L_sext, L_drift, L_corr, dL, L_lens, dL, L_dip, dL])
        
        # bending strength array
        inv_rhos = np.array([0, self.dipoleField(), 0, 0, 0, B_corr, 0, 0, 0, B_corr, 0, 0, 0, self.dipoleField(), 0]) * SI.c / self.E0
        
        # focusing strength array
        ks = np.array([0, 0, 0, g_lens, 0, 0, 0, 0, 0, 0, 0, g_lens, 0, 0, 0]) * SI.c / self.E0
        
        # sextupole strength array
        ms = np.array([0, 0, 0, 0, 0, 0, 0, m_sext, 0, 0, 0, 0, 0, 0, 0])
        
        # plasma-lens transverse taper array
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0, 0, tau_lens, 0, 0, 0])
        
        return ls, inv_rhos, ks, ms, taus
    
    
    # first half of the lattice (up to middle of the sextupole)
    def halfLattice(self, g_lens=0, B_corr=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, B_corr, m_sext, tau_lens)
        inds = range(int(np.ceil(len(ls)/2)))
        ls_half = ls[inds]
        ls_half[-1] = ls_half[-1]/2
        return ls_half, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # first quarter of the lattice (up to middle of first lens)
    def quarterLattice(self, g_lens=0, B_corr=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, B_corr, m_sext, tau_lens)
        inds = range(int(np.ceil(len(ls)/4)))
        ls_quart = ls[inds]
        ls_quart[-1] = ls_quart[-1]/2
        return ls_quart, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # total length of lattice
    def length(self):
        ls, _, _, _, _ = self.fullLattice()
        return np.sum(ls)
        
    
    # make run script file
    def makeRunScript(self):
        return CONFIG.opal_path + 'opal/apis/elegant/templates/runscript_interstage.ele'
    
    
    def makeLattice(self, beam, output_filename):
        
        # perform matching to find exact element strengths
        g_lens, tau_lens, B_corr, m_sext = self.match()
        
        # make lens field
        lensfile = elegant_apl_fieldmap2D(tau_lens)
        
        # inputs
        inputs = {'charge': abs(beam.charge()),
                  'dipole_length': self.dipoleLength(),
                  'dipole_angle': self.dipoleLength()*self.dipoleField()*SI.c/self.E0,
                  'corrector_length': self.correctorLength(),
                  'corrector_angle': self.correctorLength()*B_corr*SI.c/self.E0,
                  'lens_length': self.lensLength(),
                  'lens_filename': lensfile,
                  'lens_strength': g_lens,
                  'drift_length': self.driftLength(),
                  'spacer_length': self.spacerLength(),
                  'sextupole_length': self.sextupoleLength(),
                  'sextupole_strength': m_sext,
                  'output_filename': output_filename}

        # make lattice file from template
        lattice_template = CONFIG.opal_path + 'opal/apis/elegant/templates/lattice_interstage.lte'
        latticefile = tempfile.gettempdir() + '/interstage.lte'
        with open(lattice_template, 'r') as fin, open(latticefile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
        
        return latticefile
    
    
    # match the beta function, first- and second-order dispersions
    def match(self):
        
        # define half lattice
        ls_half, _, _, _, _ = self.halfLattice()
        inv_rhos_half = lambda B: np.array([0, self.Bdip, 0, 0, 0, B, 0, 0]) * SI.c/self.E0
        ks_half = lambda g: np.array([0, 0, 0, g, 0, 0, 0, 0]) * SI.c/self.E0
        ms_half = lambda m: np.array([0, 0, 0, 0, 0, 0, 0, m])
        taus_half = lambda tau: np.array([0, 0, 0, tau, 0, 0, 0, 0])
        
        # minimizer function for beta matching (central alpha function is zero)
        def minfun_beta(params):
            _, alpha, _ = evolveBetaFunction(ls_half, ks_half(params[0]), self.initialBetaFunction(), fast=True)         
            return alpha**2
        
        # match the beta function
        result_beta = minimize(minfun_beta, self.g_max, tol=1e-20)
        g_lens = result_beta.x[0]
        
        # minimizer function for first-order dispersion (central dispersion prime is zero)
        def minfun_Dx(p):
            _, Dpx, _ = evolveDispersion(ls_half, inv_rhos_half(p[0]), ks_half(g_lens), fast=True)         
            return Dpx**2
        
        # match the first-order dispersion
        result_Dx = minimize(minfun_Dx, 0, tol=1e-20)
        B_corr = result_Dx.x[0]
        
        # calculate the required transverse-taper gradient
        ls_quart, _, _, _, _ = self.quarterLattice()
        inv_rhos_quart = np.array([0, self.Bdip, 0, 0]) * SI.c/self.E0
        ks_quart = np.array([0, 0, 0, g_lens]) * SI.c/self.E0
        Dx_lens, _, _ = evolveDispersion(ls_quart, inv_rhos_quart, ks_quart, fast=True)
        tau_lens = 1/Dx_lens
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        def minfun_DDx(p):
            _, DDpx, _ = evolveSecondOrderDispersion(ls_half, inv_rhos_half(B_corr), ks_half(g_lens), ms_half(p[0]), taus_half(tau_lens), fast=True)
            return DDpx**2
        
        # match the second-order dispersion
        m_guess = 4*tau_lens/self.sextupoleLength()
        result_DDx = minimize(minfun_DDx, m_guess, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        m_sext = result_DDx.x[0]
        
        # plot results
        if False:
            ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, B_corr, m_sext, tau_lens)
            _, _, evolution_beta = evolveBetaFunction(ls, ks, self.initialBetaFunction())
            _, _, evolution_Dx = evolveDispersion(ls, inv_rhos, ks)
            _, _, evolution_DDx = evolveSecondOrderDispersion(ls, inv_rhos, ks, ms, taus)

            # prepare figure
            fig, axs = plt.subplots(1,3)
            fig.set_figwidth(20)
            fig.set_figheight(4)
            axs[0].plot(evolution_beta[0,:], evolution_beta[1,:])
            axs[1].plot(evolution_Dx[0,:], evolution_Dx[1,:])
            axs[1].plot(evolution_DDx[0,:], evolution_DDx[1,:])
            axs[2].plot(evolution_DDx[0,:], evolution_DDx[2,:])
        
        return g_lens, tau_lens, B_corr, m_sext
        

    # track a beam through the lattice using ELEGANT
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # environment variables
        envars = {}
        envars['ENERGY'] = self.E0 / 1e6 # [MeV]
        
        # make lattice
        beamfile = tempfile.gettempdir() + '/beam.bun'
        envars['LATTICE'] = self.makeLattice(beam0, beamfile)
        
        # run ELEGANT
        runfile = self.makeRunScript()
        beam = elegant_run(runfile, beam0, beamfile, envars, quiet=True)
        
        return super().track(beam, savedepth, runnable, verbose)
    