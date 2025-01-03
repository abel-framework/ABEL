import uuid, os, scipy, shutil, subprocess, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle
from functools import partial
from matplotlib.animation import FuncAnimation
from functools import partial
import scipy.constants as SI
from string import Template
from types import SimpleNamespace
import abel
from abel import CONFIG, Interstage, Beam
from abel.apis.elegant.elegant_api import elegant_run, elegant_apl_fieldmap2D, elegant_read_beam
from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_second_order_dispersion


class InterstageElegant(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, dipole_length=None, dipole_field=1, Bdip2=None, lens_x_offset=0.0, enable_isr=True, enable_csr=True, save_evolution=False, save_apl_field_map=False):
        
        super().__init__()
        
        self.nom_energy = nom_energy
        self.beta0 = beta0
        self.dipole_length = dipole_length
        self.dipole_field = dipole_field
        self.Bdip2 = Bdip2
        self.g_max = 1000 # [T/m] 
        self.disableNonlinearity = False
        self.lens_x_offset = lens_x_offset
        self.lens_y_offset = 0.0
        
        self.default_Bdip2_Bdip_ratio = 0.8
    
        self.enable_isr = enable_isr
        self.enable_csr = enable_csr
        self.save_evolution = save_evolution
        self.save_apl_field_map = save_apl_field_map
        self.run_path = None

        self.beam0_charge_sign = None

        self.apl_field_map = None
        self.ramp_beta_mag = None
        self.dipole_length = None
        self.dipole_angle = None
        self.chicane_dipole_length = None
        self.chicane_dipole_angle1 = None
        self.chicane_dipole_angle2 = None
        self.lens_length = None
        self.lens_strength = None
        self.tau_lens = None
        self.spacer_length = None
        self.sextupole_length = None
        self.sextupole_strength = None
        
        self.element_lengths = None
                

    def set_nom_energy(self, nom_energy):
        self.nom_energy = nom_energy

    
    def set_lens_offset(self, lens_x_offset, lens_y_offset):
        self.lens_x_offset = lens_x_offset
        self.lens_y_offset = lens_y_offset
        
    
    # evaluate beta function 
    def __eval_initial_beta_function(self):
        if callable(self.beta0):
            return self.beta0(self.nom_energy)
        else:
            return self.beta0

    
    # evaluate dipole field
    def __eval_dipole_field(self):
        if callable(self.dipole_field):
            return self.dipole_field(self.nom_energy)
        else:
            return self.dipole_field

    
    # evaluate dipole length  
    def __eval_dipole_length(self):
        if callable(self.dipole_length):
            return self.dipole_length(self.nom_energy)
        else:
            return self.dipole_length

    
    # spacer length
    def __eval_spacer_length(self):
        return 0.05 * self.__eval_dipole_length()

    
    # plasma-lens length
    def __eval_lens_length(self):
        f = (self.__eval_dipole_length()+2*self.__eval_spacer_length())/2
        k = self.g_max*SI.c/self.nom_energy
        return 1/(k*f)

    
    # chicane dipole length
    def __eval_chicane_dipole_length(self):
        return 0.6 * self.__eval_dipole_length()

    
    # evaluate dipole field (or use default)
    def __eval_chicane_field(self):
        if self.Bdip2 is None:
            return self.__eval_dipole_field() * self.default_Bdip2_Bdip_ratio
        elif callable(self.Bdip2):
            return self.Bdip2(self.nom_energy)
        else:
            return self.Bdip2

    
    # sextupole length
    def __eval_sextupole_length(self):
        return 0.4 * self.__eval_dipole_length()

    
    # full lattice 
    def __full_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        
        # element length array
        dL = self.__eval_spacer_length()
        L_dip = self.__eval_dipole_length()
        L_lens = self.__eval_lens_length()
        L_chic = self.__eval_chicane_dipole_length()
        L_sext = self.__eval_sextupole_length()
        ls = np.array([dL, L_dip, dL, L_lens, dL, L_chic, dL, L_chic, dL, L_sext, dL, L_chic, dL, L_chic, dL, L_lens, dL, L_dip, dL])
        self.element_lengths = ls
        
        # bending strength array
        Bdip = self.__eval_dipole_field()
        Bdip2 = self.__eval_chicane_field()
        inv_rhos = np.array([0, Bdip, 0, 0, 0, Bdip2, 0, Bdip3, 0, 0, 0, Bdip3, 0, Bdip2, 0, 0, 0, Bdip, 0]) * SI.c / self.nom_energy
        
        # focusing strength array
        ks = np.array([0, 0, 0, g_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g_lens, 0, 0, 0]) * SI.c / self.nom_energy
        
        # sextupole strength array
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # plasma-lens transverse taper array
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tau_lens, 0, 0, 0])
        
        return ls, inv_rhos, ks, ms, taus
    
    
    # first half of the lattice (up to middle of the sextupole)
    def __half_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(int(np.ceil(len(ls)/2)))
        ls_half = ls[inds]
        ls_half[-1] = ls_half[-1]/2
        return ls_half, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # first quarter of the lattice (up to middle of first lens)
    def __quarter_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(4)
        ls_quart = ls[inds]
        ls_quart[-1] = ls_quart[-1]/2
        return ls_quart, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # total length of lattice
    def get_length(self):
        ls, _, _, _, _ = self.__full_lattice()
        return np.sum(ls)


    def __make_run_script(self, latticefile, inputbeamfile, tmpfolder=None):

        # create temporary CSV file and folder
        make_new_tmpfolder = tmpfolder is None
        if make_new_tmpfolder:
            tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
            os.mkdir(tmpfolder)
        tmpfile = tmpfolder + '/runfile.ele'

        # inputs
        path_beam_centroid_file = tmpfolder + '/centroid_vs_s.cen'  # Used to output a file in tmpfolder containing data of the beam centroids as a function of s.
        path_twiss_parameter_file =  tmpfolder + '/twiss_vs_s.twi'  # Used to output a file in tmpfolder containing data of the beam centroids as a function of s.
        inputs = {'p_central_mev': self.nom_energy/1e6,
                  'latticefile': latticefile,
                  'inputbeamfile': inputbeamfile,
                  'path_to_beam_centroid_file': path_beam_centroid_file,
                  'path_to_uncoupled_Twiss_parameter_output_file': path_twiss_parameter_file}

        runfile_template = os.path.join(os.path.dirname(abel.apis.elegant.elegant_api.__file__), 'templates', 'runscript_interstage.ele')
        with open(runfile_template, 'r') as fin, open(tmpfile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)

        return tmpfile
    
    
    def __make_lattice(self, beam, outputbeamfile, latticefile, lensfile, evolutionfolder, save_evolution=False, tmpfolder=None):
        
        # perform matching to find exact element strengths
        g_lens, tau_lens, Bdip3, m_sext = self.match()

        self.dipole_length = self.__eval_dipole_length()
        self.dipole_angle = self.__eval_dipole_length() * self.__eval_dipole_field() * SI.c/self.nom_energy
        self.chicane_dipole_length = self.__eval_chicane_dipole_length()
        self.chicane_dipole_angle1 = self.__eval_chicane_dipole_length() * self.__eval_chicane_field() * SI.c/self.nom_energy
        self.chicane_dipole_angle2 = self.__eval_chicane_dipole_length() * Bdip3 * SI.c/self.nom_energy
        self.lens_length = self.__eval_lens_length()
        self.lens_strength = g_lens
        self.tau_lens = tau_lens
        self.spacer_length = self.__eval_spacer_length()
        self.sextupole_length = self.__eval_sextupole_length()
        self.sextupole_strength = m_sext
        
        if self.disableNonlinearity:
            tau_lens = 0
            m_sext = 0
        
        # make lens field
        elegant_apl_fieldmap2D(tau_lens, lensfile, lens_x_offset=self.lens_x_offset, lens_y_offset=self.lens_y_offset, tmpfolder=tmpfolder)

        if self.save_apl_field_map:
            file = open(tmpfolder + '/Bmap.csv', "rb")
            data = np.loadtxt(file, delimiter=',')  # Avoids extracting the columns containing strings such as ElementName.
            file.close()
            self.apl_field_map = data
        
        # make lattice file from template
        lattice_template = os.path.join(os.path.dirname(abel.apis.elegant.elegant_api.__file__), 'templates', 'lattice_interstage.lte')

        if save_evolution:
            num_watches = 5
            watch_disabled = False
        else:
            num_watches = 1
            watch_disabled = True
            
        inputs = {'charge': abs(beam.charge()),
                  'num_kicks': int(100/num_watches),
                  'dipole_length': self.__eval_dipole_length()/num_watches,
                  'dipole_angle': self.__eval_dipole_length()*self.__eval_dipole_field()*SI.c/self.nom_energy/num_watches,
                  'chicanedipole_length': self.__eval_chicane_dipole_length()/num_watches,
                  'chicanedipole_angle1': self.__eval_chicane_dipole_length()*self.__eval_chicane_field() * SI.c/self.nom_energy/num_watches,
                  'chicanedipole_angle2': self.__eval_chicane_dipole_length()*Bdip3*SI.c/self.nom_energy/num_watches,
                  'lens_length': self.__eval_lens_length()/num_watches,
                  'lens_filename': lensfile,
                  'lens_strength': g_lens,
                  'spacer_length': self.__eval_spacer_length()/num_watches,
                  'sextupole_length': self.__eval_sextupole_length()/num_watches,
                  'sextupole_strength': m_sext,
                  'output_filename': outputbeamfile,
                  'enable_ISR': int(self.enable_isr),
                  'enable_CSR': int(self.enable_csr),
                  'num_out': int(num_watches),
                  'watch_filename': evolutionfolder + 'output_%03ld.bun',
                  'watch_disabled': int(watch_disabled)}
        
        with open(lattice_template, 'r') as fin, open(latticefile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
        
    # match the beta function, first- and second-order dispersions
    def match(self, make_plot=False):
        
        # define half lattice
        ls_half, _, _, _, _ = self.__half_lattice()
        Bdip = self.__eval_dipole_field()
        Bdip2 = self.__eval_chicane_field()
        inv_rhos_half = lambda B3: np.array([0, Bdip, 0, 0, 0, Bdip2, 0, B3, 0, 0]) * SI.c/self.nom_energy
        ks_half = lambda g: np.array([0, 0, 0, g, 0, 0, 0, 0, 0, 0]) * SI.c/self.nom_energy
        ms_half = lambda m: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m])
        taus_half = lambda tau: np.array([0, 0, 0, tau, 0, 0, 0, 0, 0, 0])
        
        # minimizer function for beta matching (central alpha function is zero)
        def minfun_beta(params):
            _, alpha, _ = evolve_beta_function(ls_half, ks_half(params[0]), self.__eval_initial_beta_function(), fast=True)         
            return alpha**2
        
        # match the beta function
        result_beta = scipy.optimize.minimize(minfun_beta, self.g_max, tol=1e-20)
        g_lens = result_beta.x[0]
        
        # minimizer function for first-order dispersion (central dispersion prime is zero)
        def minfun_Dx(p):
            _, Dpx, _ = evolve_dispersion(ls_half, inv_rhos_half(p[0]), ks_half(g_lens), fast=True)         
            return Dpx**2
        
        # match the first-order dispersion
        Bdip3_guess = -Bdip2;
        result_Dx = scipy.optimize.minimize(minfun_Dx, Bdip3_guess, tol=1e-20)
        Bdip3 = result_Dx.x[0]
        
        # calculate the required transverse-taper gradient
        ls_quart, _, _, _, _ = self.__quarter_lattice()
        inv_rhos_quart = np.array([0, Bdip, 0, 0]) * SI.c/self.nom_energy
        ks_quart = np.array([0, 0, 0, g_lens]) * SI.c/self.nom_energy
        Dx_lens, _, _ = evolve_dispersion(ls_quart, inv_rhos_quart, ks_quart, fast=True)
        tau_lens = 1/Dx_lens
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        def minfun_DDx(p):
            _, DDpx, _ = evolve_second_order_dispersion(ls_half, inv_rhos_half(Bdip3), ks_half(g_lens), ms_half(p[0]), taus_half(tau_lens), fast=True)
            return DDpx**2
        
        # match the second-order dispersion
        m_guess = 4*tau_lens/self.__eval_sextupole_length()
        result_DDx = scipy.optimize.minimize(minfun_DDx, m_guess, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        m_sext = result_DDx.x[0]
        
        # plot results
        if make_plot:
            ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
            _, _, evolution_beta = evolve_beta_function(ls, ks, self.__eval_initial_beta_function())
            _, _, evolution_Dx = evolve_dispersion(ls, inv_rhos, ks)
            _, _, evolution_DDx = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus)

            # prepare figure
            fig, axs = plt.subplots(1,3)
            fig.set_figwidth(20)
            fig.set_figheight(4)
            axs[0].plot(evolution_beta[0,:], evolution_beta[1,:])
            axs[0].set_xlabel(r'$s$ [m]')
            axs[0].set_ylabel(r'$\beta$ [m]')
            axs[1].plot(evolution_Dx[0,:], evolution_Dx[1,:], label='evolution_Dx[1,:]')
            axs[1].plot(evolution_DDx[0,:], evolution_DDx[1,:], label='evolution_DDx[1,:]')
            axs[1].set_xlabel(r'$s$ [m]')
            axs[1].set_ylabel('First order dispersion [m]')
            axs[2].plot(evolution_DDx[0,:], evolution_DDx[2,:])
            axs[2].set_xlabel(r'$s$ [m]')
            axs[2].set_ylabel('Second order dispersion [m]')

            #print('evolution_beta[1,0]-evolution_beta[1,-1]: ', evolution_beta[1,0]-evolution_beta[1,-1])
            #print('evolution_beta[1,0]: ', evolution_beta[1,0])
            #print('max(evolution_beta[1,:]): ', np.max(evolution_beta[1,:]))                                ########################################### to be del
            #print('evolution_Dx[1,0]: ', evolution_Dx[1,0])                                                   ###############################################
            #print('evolution_Dx[1,-1]: ', evolution_Dx[1,-1])                                                   ###############################################
            #print('evolution_DDx[1,0]: ', evolution_DDx[1,0])                                                   ###############################################
            #print('evolution_DDx[1,-1]: ', evolution_DDx[1,-1])                                                   ###############################################
            #print('evolution_DDx[2,0]: ', evolution_DDx[2,0])                                                   ###############################################
            #print('evolution_DDx[2,-1]: ', evolution_DDx[2,-1])                                                 ###############################################
        
        return g_lens, tau_lens, Bdip3, m_sext
        

    # track a beam through the lattice using ELEGANT
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        #print('Start of interstage')
        #print('beam0.x_offset:', beam0.x_offset())                                                 ############################################### to be del
        #print('beam0.y_offset:', beam0.y_offset(), '\n')                                           ############################################### to be del
        
        # make temporary folder and files
        parent_dir = CONFIG.temp_path
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the temporary folder
        tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
        os.mkdir(tmpfolder)
        inputbeamfile = tmpfolder + 'input_beam.bun'
        outputbeamfile = tmpfolder + 'output_beam.bun'
        latticefile = tmpfolder + 'interstage.lte'
        lensfile = tmpfolder + 'map.csv'
        evolution_folder = tmpfolder + 'evolution' + os.sep
        os.mkdir(evolution_folder)
        
        # make lattice file
        self.__make_lattice(beam0, outputbeamfile, latticefile, lensfile, evolution_folder, self.save_evolution, tmpfolder=tmpfolder)
        
        # run ELEGANT
        runfile = self.__make_run_script(latticefile, inputbeamfile, tmpfolder)
        beam = elegant_run(runfile, beam0, inputbeamfile, outputbeamfile, quiet=True, tmpfolder=tmpfolder)
        self.beam0_charge_sign = beam0.charge_sign()  # patch due to ELEGANT not setting correct charge sign

        # save evolution of beam parameters
        if self.save_evolution:
            self.__extract_evolution(tmpfolder, evolution_folder, runnable)
            if os.path.isdir(self.run_path + 'evolution' + os.sep):
                shutil.rmtree(self.run_path + 'evolution' + os.sep)  # Delete the existing folder before copying.
            shutil.copytree(evolution_folder, self.run_path + 'evolution' + os.sep)

        # clean extreme outliers
        beam.remove_halo_particles()
        
        # remove temporary files
        shutil.rmtree(tmpfolder)

        #print('End of interstage')
        #print('beam.x_offset-beam0.x_offset:', beam.x_offset()-beam0.x_offset())          ############################################### to be del
        #print('beam.y_offset-beam0.y_offset:', beam.y_offset()-beam0.y_offset())          ############################################### to be del

        return super().track(beam, savedepth, runnable, verbose)


    # Extract the evolution of various beam parameters as a function of s.
    def __extract_evolution(self, tmpfolder, evolution_folder, runnable):
        insitu_path = tmpfolder + 'diags/insitu/'
        
        # Run system command for converting .cen file to a .csv file
        cmd = CONFIG.elegant_exec + '/sdds2stream ' + tmpfolder + '/centroid_vs_s.cen -columns=s,Cx,Cy,Cxp,Cyp,Cs,Cdelta,Particles,pCentral,Charge,ElementName,ElementType,ElementOccurence >' + tmpfolder + '/centroids.csv'
        subprocess.call(cmd, shell=True)

        # Load data from .csv file. All quantities in SI units unless otherwise specified.
        stop_column = 10
        file = open(tmpfolder + '/centroids.csv', "rb")
        data = np.loadtxt(file, delimiter=' ', usecols=range(stop_column))  # Avoids extracting the columns containing strings such as ElementName.
        file.close()
        
        s = data[:,0]
        x_offset = data[:,1]
        y_offset = data[:,2]
        x_angle = data[:,3]
        y_angle = data[:,4]
        s_offset = data[:,5]
        #rel_momentum_deviation = data[:,6]  # (p-p0)/p0. Small differences from energy offset calcualted using the beam energy directly.
        num_particles = data[:,7]
        mean_gamma = data[:,8] # Reference beta*gamma. Dimensionless?
        #self.evolution.charge = data[:,9]  # Wrong sign due to the CHARGE element in the beginning of the lattice.

        self.evolution.location_centroids = s
        self.evolution.x_offset = x_offset
        self.evolution.y_offset = y_offset
        self.evolution.x_angle = x_angle
        self.evolution.y_angle = y_angle
        self.evolution.s_offset = s_offset
        #self.evolution.rel_momentum_deviation = rel_momentum_deviation
        self.evolution.num_particles = num_particles
        
        #self.evolution.energy = mean_gamma*SI.m_e*SI.c**2/SI.e  # [eV]
        self.evolution.energy = np.empty([0])  # [eV]
    
        self.evolution.location = np.empty([0])
        #self.evolution.x_offset_beam = np.empty([0])
        #self.evolution.y_offset_beam = np.empty([0])
        #self.evolution.xp_offset_beam = np.empty([0])
        #self.evolution.yp_offset_beam = np.empty([0])
        #self.evolution.z_offset = np.empty([0]))  # No need to use this, as ELEGANT can output this faster.
        self.evolution.beam_size_x = np.empty([0])
        self.evolution.divergence_x = np.empty([0])
        self.evolution.beam_size_y = np.empty([0])
        self.evolution.bunch_length = np.empty([0])
        self.evolution.norm_emittance_x = np.empty([0])
        self.evolution.norm_emittance_y = np.empty([0])
        self.evolution.rel_energy_spread = np.empty([0])
        for file in sorted(os.listdir(evolution_folder)):
            output_beam = elegant_read_beam(evolution_folder + os.fsdecode(file), tmpfolder=tmpfolder)
            self.evolution.location = np.append(self.evolution.location, output_beam.location)
            #self.evolution.x_offset_beam = np.append(self.evolution.x_offset_beam, output_beam.x_offset())  # No need to use this, as ELEGANT can output this faster.
            #self.evolution.y_offset_beam = np.append(self.evolution.y_offset_beam, output_beam.y_offset())  # No need to use this, as ELEGANT can output this faster.
            #self.evolution.xp_offset_beam = np.append(self.evolution.xp_offset_beam, output_beam.x_angle())  # No need to use this, as ELEGANT can output this faster.
            #self.evolution.yp_offset_beam = np.append(self.evolution.yp_offset_beam, output_beam.y_angle())  # No need to use this, as ELEGANT can output this faster.
            #self.evolution.z_offset = np.append(self.evolution.z_offset, output_beam.z_offset())  # No need to use this, as ELEGANT can output this faster.
            self.evolution.beam_size_x = np.append(self.evolution.beam_size_x, output_beam.beam_size_x())
            self.evolution.divergence_x = np.append(self.evolution.divergence_x, output_beam.divergence_x())
            self.evolution.beam_size_y = np.append(self.evolution.beam_size_y, output_beam.beam_size_y())
            self.evolution.bunch_length = np.append(self.evolution.bunch_length, output_beam.bunch_length())
            self.evolution.norm_emittance_x = np.append(self.evolution.norm_emittance_x, output_beam.norm_emittance_x())
            self.evolution.norm_emittance_y = np.append(self.evolution.norm_emittance_y, output_beam.norm_emittance_y())
            self.evolution.rel_energy_spread = np.append(self.evolution.rel_energy_spread, output_beam.rel_energy_spread())
            self.evolution.energy = np.append(self.evolution.energy, output_beam.energy())
        
        # Check the offsets using output beams.
        # Delete centroids.csv... may already have been taken care of in track()?
    
        
    # Plot the evolution of various beam parameters as a function of s.
    def plot_evolution(self):

        nom_energy = self.nom_energy
        s_centroids = self.evolution.location_centroids
        #rel_momentum_deviation = self.evolution.rel_momentum_deviation
        num_particles = self.evolution.num_particles
        s_offset = self.evolution.s_offset
        x_offset = self.evolution.x_offset
        y_offset = self.evolution.y_offset
        x_angle = self.evolution.x_angle
        y_angle = self.evolution.y_angle

        #x_offset_beam = self.evolution.x_offset_beam  # No need to use this, as ELEGANT can output this faster.
        #y_offset_beam = self.evolution.y_offset_beam  # No need to use this, as ELEGANT can output this faster.
        #xp_offset_beam = self.evolution.xp_offset_beam  # No need to use this, as ELEGANT can output this faster.
        #yp_offset_beam = self.evolution.yp_offset_beam  # No need to use this, as ELEGANT can output this faster.
        energy = self.evolution.energy

        s = self.evolution.location
        #z_offset = self.evolution.z_offset)  # No need to use this, as ELEGANT can output this faster.
        norm_emittance_xs = self.evolution.norm_emittance_x
        norm_emittance_ys = self.evolution.norm_emittance_y
        beam_size_xs = self.evolution.beam_size_x
        beam_size_ys = self.evolution.beam_size_y
        bunch_lengths = self.evolution.bunch_length
        rel_energy_spreads = self.evolution.rel_energy_spread
        rel_energy_offset = energy/nom_energy-1

        long_label = '$s$ [m]'

        # line format
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        #af = 0.2
        
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)

        #axs[0,0].plot(s_centroids, np.ones(len(norm_energy))*nom_energy/1e9, ':', color=col0)
        #axs[0,0].plot(s_centroids, energy/1e9, color=col1)
        #axs[0,0].set_xlabel(long_label)
        #axs[0,0].set_ylabel('Energy [GeV]')

        axs[0,0].plot(s_centroids, (s_centroids-s_offset)*1e6, color=col1, marker='x')
        #axs[0,0].plot(s, z_offset*1e6, color='red')  # No need to use this, as ELEGANT can output this faster.
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_ylabel(r'Longitudinal offset [$\mathrm{\mu}$m]')

        axs[1,0].plot(s, rel_energy_spreads*100, color=col1)
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_ylabel('Energy spread [%]')
        axs[1,0].set_yscale('log')

        #axs[2,0].plot(s_centroids, np.zeros(len(rel_momentum_deviation)), ':', color=col0)
        #axs[2,0].plot(s_centroids, rel_momentum_deviation*100, color=col1, marker='x')
        #axs[2,0].set_xlabel(long_label)
        #axs[2,0].set_ylabel('Relative momentum deviation [%]')

        #axs[2,0].plot(s_centroids, rel_momentum_deviation*100, color='red', marker='x')
        axs[2,0].plot(s, np.zeros(len(rel_energy_offset)), ':', color=col0)
        axs[2,0].plot(s, rel_energy_offset*100, color=col1)
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_ylabel('Energy offset [%]')

        axs[0,1].plot(s_centroids, num_particles[0]*np.ones(num_particles.shape), ':', color=col0)
        axs[0,1].plot(s_centroids, num_particles, color=col1)
        axs[0,1].set_xlabel(long_label)
        axs[0,1].set_ylabel('Number of macro particles')

        axs[1,1].plot(s, bunch_lengths*1e6, color=col1)
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_ylabel(r'Bunch length [$\mathrm{\mu}$m]')

        axs[2,1].plot(s_centroids, np.zeros(x_angle.shape), ':', color=col0)
        axs[2,1].plot(s_centroids, x_angle*1e6, color=col1, marker='x', label=r'$\langle x\' \rangle$')
        axs[2,1].plot(s_centroids, y_angle*1e6, color=col2, marker='x', label=r'$\langle y\' \rangle$')
        #axs[2,1].plot(s, xp_offset_beam*1e6, color='red')
        #axs[2,1].plot(s, yp_offset_beam*1e6, color='black')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_ylabel(r'Angular offset [$\mathrm{\mu}$rad]')
        axs[2,1].legend()

        axs[0,2].plot(s, np.ones(len(norm_emittance_xs))*norm_emittance_xs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[0,2].plot(s, np.ones(len(norm_emittance_ys))*norm_emittance_ys[0]*1e6, ':', color=col0)
        axs[0,2].plot(s, norm_emittance_xs*1e6, color=col1, marker='x', label=r'$\varepsilon_{\mathrm{n}x}$')
        axs[0,2].plot(s, norm_emittance_ys*1e6, color=col2, marker='x', label=r'$\varepsilon_{\mathrm{n}y}$')
        axs[0,2].set_xlabel(long_label)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_yscale('log')
        axs[0,2].legend()

        #axs[1,2].plot(s, (Es_nom[0]/Es_nom)**(1/4)*beam_size_xs[0]*1e6, ':', color=col0, label='Nominal value')
        #axs[1,2].plot(s, (Es_nom[0]/Es_nom)**(1/4)*beam_size_ys[0]*1e6, ':', color=col0)
        axs[1,2].plot(s, np.ones(beam_size_xs.shape)*beam_size_xs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[1,2].plot(s, np.ones(beam_size_ys.shape)*beam_size_ys[0]*1e6, ':', color=col0)
        axs[1,2].plot(s, beam_size_xs*1e6, color=col1, label=r'$\sigma_x$')
        axs[1,2].plot(s, beam_size_ys*1e6, color=col2, label=r'$\sigma_y$')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[1,2].set_yscale('log')
        axs[1,2].legend()

        axs[2,2].plot(s_centroids, np.zeros(x_offset.shape), ':', color=col0)
        axs[2,2].plot(s_centroids, x_offset*1e6, color=col1, marker='x', label=r'$\langle x \rangle$')
        axs[2,2].plot(s_centroids, y_offset*1e6, color=col2, marker='x', label=r'$\langle y \rangle$')
        #axs[2,2].plot(s, x_offset_beam*1e6, color='red')
        #axs[2,2].plot(s, y_offset_beam*1e6, color='black')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        #axs[2,2].set_yscale('log')
        axs[2,2].legend()


    # Extract the positions of the lattice elements...


# ==================================================
    def print_summary(self, drive_beam, initial_main_beam, beam_out):
        with open(self.run_path + 'output.txt', 'w') as f:
            print('================================================', file=f)
            print(f"Ramp beta magnification:\t\t {self.ramp_beta_mag :.3f}", file=f)
            
            if callable(self.dipole_field):
                dipole_field = self.dipole_field(initial_main_beam.energy())
                print(f"Dipole field [T]:\t\t {dipole_field :.3f}", file=f)
            else:
                dipole_field = self.dipole_field
                print(f"Dipole field [T]:\t\t\t {dipole_field :.3f}", file=f)
    
            if self.dipole_length is None:
                print(f"Dipole arc length [m]:\t {'Not registered.' :s}", file=f)
            else:
                dipole_length = self.dipole_length
                print(f"Dipole arc length [m]:\t\t\t {dipole_length :.3f}", file=f)
    
            if self.dipole_angle is None:
                print(f"Dipole angle [rad]:\t {'Not registered.' :s}", file=f)
            else:
                print(f"Dipole angle [rad]:\t\t\t {self.dipole_angle :.3f}", file=f)
    
            if self.chicane_dipole_length is None:
                print(f"Chicane dipole arc length [m]:\t {'Not registered.' :s}", file=f)
            else:
                print(f"Chicane dipole arc length [m]:\t\t {self.chicane_dipole_length :.3f}", file=f)
    
            if self.chicane_dipole_angle1 is None:
                print(f"Chicane dipole angle 1 [rad]:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Chicane dipole angle 1 [rad]:\t\t {self.chicane_dipole_angle1 :.3f}", file=f)
    
            if self.chicane_dipole_angle2 is None:
                print(f"Chicane dipole angle 2 [rad]:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Chicane dipole angle 2 [rad]:\t\t {self.chicane_dipole_angle2 :.3f}", file=f)
    
            if self.lens_length is None:
                print(f"Lens length:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Lens length:\t\t\t\t {self.lens_length :.3f}", file=f)
    
            if self.lens_strength is None:
                print(f"Lens strength:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Lens strength:\t\t\t\t {self.lens_strength :.3f}", file=f)
    
            if self.tau_lens is None:
                print(f"Lens transverse-taper gradient:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Lens transverse-taper gradient:\t\t {self.tau_lens :.3f}", file=f)
    
            if self.lens_x_offset is None:
                print(f"Lens x-offset [um]:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Lens x-offset [um]:\t\t\t {self.lens_x_offset*1e6 :.3f}", file=f)
    
            if self.spacer_length is None:
                print(f"Spacer length:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Spacer length:\t\t\t\t {self.spacer_length :.3f}", file=f)
    
            if self.sextupole_length is None:
                print(f"Sextupole length:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Sextupole length:\t\t\t {self.sextupole_length :.3f}", file=f)
    
            if self.sextupole_strength is None:
                print(f"Sextupole strength:\t\t {'Not registered.' :s}", file=f)
            else:
                print(f"Sextupole strength:\t\t\t {self.sextupole_strength :.3f}", file=f)
                
            
            print(f"ISR enabled:\t\t\t\t {str(self.enable_isr) :s}", file=f)
            print(f"CSR enabled:\t\t\t\t {str(self.enable_csr) :s}", file=f)
            print('------------------------------------------------\n', file=f)
    
            
            print('-------------------------------------------------------------------------------------', file=f)
            print('Quantity \t\t\t\t\t Drive beam \t\t Main beam', file=f)
            print('-------------------------------------------------------------------------------------', file=f)
            print(f"Initial number of macroparticles:\t\t {len(drive_beam.xs()) :d}\t\t\t {len(initial_main_beam.xs()) :d}", file=f)
            print(f"Initial beam population:\t\t\t {(np.sum(drive_beam.weightings())) :.3e} \t\t {(np.sum(initial_main_beam.weightings())) :.3e}\n", file=f)
            
            print(f"Initial mean gamma:\t\t\t\t {drive_beam.gamma() :.3f} \t\t {initial_main_beam.gamma() :.3f}", file=f)
            print(f"Initial mean energy [GeV]:\t\t\t {drive_beam.energy()/1e9 :.3f} \t\t {initial_main_beam.energy()/1e9 :.3f}", file=f)
            print(f"Initial rms energy spread [%]:\t\t\t {drive_beam.rel_energy_spread()*1e2 :.3f} \t\t\t {initial_main_beam.rel_energy_spread()*1e2 :.3f}", file=f)
            print(f"Final rms energy spread [%]:\t\t\t  \t\t\t {beam_out.rel_energy_spread()*1e2 :.3f}\n", file=f)
    
            print(f"Initial beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f} \t\t {initial_main_beam.x_offset()*1e6 :.3f}", file=f)
            print(f"Final beam x offset [um]:\t\t\t  \t\t\t {beam_out.x_offset()*1e6 :.3f}", file=f)
            print(f"Initial beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f} \t\t\t {initial_main_beam.y_offset()*1e6 :.3f}", file=f)
            print(f"Final beam y offset [um]:\t\t\t  \t\t\t {beam_out.y_offset()*1e6 :.3f}", file=f)
            print(f"Initial beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f} \t\t {initial_main_beam.z_offset()*1e6 :.3f}", file=f)
            print(f"Final beam z offset [um]:\t\t\t  \t\t\t {beam_out.z_offset()*1e6 :.3f}\n", file=f)
    
            print(f"Initial beam x angular offset [urad]:\t\t {drive_beam.x_angle()*1e6 :.3f} \t\t\t {initial_main_beam.x_angle()*1e6 :.3f}", file=f)
            print(f"Final beam x angular offset [urad]:\t\t  \t\t\t {beam_out.x_angle()*1e6 :.3f}", file=f)
            print(f"Initial beam y angular offset [urad]:\t\t {drive_beam.y_angle()*1e6 :.3f} \t\t {initial_main_beam.y_angle()*1e6 :.3f}", file=f)
            print(f"Final beam y angular offset [urad]:\t\t  \t\t\t {beam_out.y_angle()*1e6 :.3f}\n", file=f)
    
            print(f"Initial normalised x emittance [mm mrad]:\t {drive_beam.norm_emittance_x()*1e6 :.3f} \t\t\t {initial_main_beam.norm_emittance_x()*1e6 :.3f}", file=f)
            print(f"Final normalised x emittance [mm mrad]:\t\t  \t\t\t {beam_out.norm_emittance_x()*1e6 :.3f}", file=f)
            print(f"Initial normalised y emittance [mm mrad]:\t {drive_beam.norm_emittance_y()*1e6 :.3f} \t\t {initial_main_beam.norm_emittance_y()*1e6 :.3f}", file=f)
            print(f"Final normalised y emittance [mm mrad]:\t\t  \t\t\t {beam_out.norm_emittance_y()*1e6 :.3f}", file=f)
            print(f"Initial angular momentum [mm mrad]:\t\t {drive_beam.angular_momentum()*1e6 :.3f} \t\t {initial_main_beam.angular_momentum()*1e6 :.3f}", file=f)
            print(f"Final angular momentum [mm mrad]:\t\t\t  \t\t {beam_out.angular_momentum()*1e6 :.3f}\n", file=f)
            
            #print(f"Initial matched beta function [mm]:\t\t\t      {self.beta0*1e3 :.3f}")
            print(f"Initial x beta function [mm]:\t\t\t {drive_beam.beta_x()*1e3 :.3f} \t\t {initial_main_beam.beta_x()*1e3 :.3f}", file=f)
            print(f"Final x beta function [mm]:\t\t\t\t  \t\t {beam_out.beta_x()*1e3 :.3f}", file=f)
            print(f"Initial y beta function [mm]:\t\t\t {drive_beam.beta_y()*1e3 :.3f} \t\t {initial_main_beam.beta_y()*1e3 :.3f}", file=f)
            print(f"Final y beta function [mm]:\t\t\t\t  \t\t {beam_out.beta_y()*1e3 :.3f}\n", file=f)
    
            print(f"Initial x beam size [um]:\t\t\t {drive_beam.beam_size_x()*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_x()*1e6 :.3f}", file=f)
            print(f"Final x beam size [um]:\t\t\t\t  \t\t\t {beam_out.beam_size_x()*1e6 :.3f}", file=f)
            print(f"Initial y beam size [um]:\t\t\t {drive_beam.beam_size_y()*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_y()*1e6 :.3f}", file=f)
            print(f"Final y beam size [um]:\t\t\t\t  \t\t\t {beam_out.beam_size_y()*1e6 :.3f}", file=f)
            print(f"Initial rms beam length [um]:\t\t\t {drive_beam.bunch_length()*1e6 :.3f} \t\t {initial_main_beam.bunch_length()*1e6 :.3f}", file=f)
            print(f"Final rms beam length [um]:\t\t\t\t  \t\t {beam_out.bunch_length()*1e6 :.3f}", file=f)
            print(f"Initial peak current [kA]:\t\t\t {drive_beam.peak_current()/1e3 :.3f} \t\t {initial_main_beam.peak_current()/1e3 :.3f}", file=f)
            print(f"Final peak current [kA]:\t\t\t\t  \t\t {beam_out.peak_current()/1e3 :.3f}", file=f)
            print('-------------------------------------------------------------------------------------', file=f)
        f.close() # Close the file

        with open(self.run_path + 'output.txt', 'r') as f:
            print(f.read())
        f.close()


    # ==================================================
    # Plot the field map of the plasma lens
    def plot_apl_field_map(self, skip_point=10):
        
        Bmap = self.apl_field_map
        
        xs_plotting = Bmap[:,0]*1e3  # [mm]
        ys_plotting = Bmap[:,1]*1e3  # [mm]
        Bxs_plotting = Bmap[:,2]
        Bys_plotting = Bmap[:,3]
        
        fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained', figsize=(9*2, 8*1))

        # Create a quiver plot
        axs[0].quiver(xs_plotting[::skip_point], ys_plotting[::skip_point], Bxs_plotting[::skip_point], Bys_plotting[::skip_point])
        axs[0].set_xlabel('$x$ [mm]')
        axs[0].set_ylabel('$y$ [mm]')

        # Create a map for B field strength
        Bs = np.sqrt(Bxs_plotting**2+Bys_plotting**2)*1e3  # [mT]
        levels = np.arange(0. , Bs.max(), 0.1)
        p=axs[1].tricontourf(xs_plotting, ys_plotting, Bs, levels=levels, cmap=CONFIG.default_cmap)
        axs[1].set_xlabel('$x$ [mm]')
        axs[1].set_ylabel('$y$ [mm]')
        cbar = plt.colorbar(p, ax=axs[1], cax=None)
        cbar.set_label('$B$ [T]')


    # Animate the horizontal sideview (top view)
    def animate_sideview_x(self, evolution_folder):

        files = sorted(os.listdir(evolution_folder))
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam_init.set_qs(self.beam0_charge_sign*beam_init.qs())      #########
        
        max_sig_index = np.argmax(self.evolution.beam_size_x)
        max_sig_beam = elegant_read_beam(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdzdx0, zs0, xs0 = max_sig_beam.phase_space_density(max_sig_beam.zs, max_sig_beam.xs)
        dQdx0, _ = max_sig_beam.projected_density(max_sig_beam.xs, bins=xs0)
        Is0, _ = max_sig_beam.current_profile(bins=zs0/SI.c)
        
        # get final beam
        beam_final = elegant_read_beam(evolution_folder + os.fsdecode(files[-1]))
        beam_final.set_qs(self.beam0_charge_sign*beam_final.qs())      #########
        dQdzdx_final, zs_final, xs_final = beam_final.phase_space_density(beam_final.zs, beam_final.xs)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        
        # prepare centroid arrays
        x0s = []
        z0s = []
        Emeans = []
        sigzs = []
        sigxs = []
        ss = []
        emitns = []
        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
                
        # frame function
        def frameFcn(i, file_list):
                    
            # get beam for this frame
            beam = elegant_read_beam(evolution_folder + os.fsdecode(file_list[i]))
            beam.set_qs(self.beam0_charge_sign*beam.qs())      #########
            
            # plot mean energy evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in interstage [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            #axs[0,0].set_ylim(0,beam_final.energy()*1.1e-9)
            axs[0,0].set_ylim(beam_init.energy()*0.9e-9, beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')

            # Plot lattice elements
            if i == 0:
                self.plot_element_lengths(ypos=0.0, height=0.15, ax=axs[0,0].twinx())
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_x()) # TODO: update to normalized amplitude
            sigzs.append(beam.bunch_length())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(min([min(emitns)*0.9e6, beam_final.norm_emittance_x()*0.8e6]), max([max(emitns)*1.1e6, emitns[0]*1.2e6]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, beam_final.bunch_length()*0.8e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdx, zs, xs = beam.phase_space_density(beam.zs, beam.xs, hbins=zs0, vbins=xs0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, xs*1e6, -dQdzdx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse offset, $x$ [$\mathrm{\mu}$m]")
            axs[1,0].set_title('Horizontal sideview (top view)')
            
            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot position projection
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, np.concatenate((xs2, np.flip(xs2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdx*1e3, xs2*1e6, color=col1)
            axs[1,1].set_ylim([min(xs0)*1e6, max(xs0)*1e6])
            axs[1,1].set_xlim([0, max([max(-dQdx), max(-dQdx_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dx$ [nC/$\mathrm{\mu}$m]")
            axs[1,1].set_ylabel(r"$x$ [$\mathrm{\mu}$m]")
            
            # plot centroid evolution
            z0s.append(beam.z_offset())
            #sigxs.append(beam.beam_size_x())
            x0s.append(beam.x_offset())
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(x0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, x0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'$x$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_xlim(min([min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, max([max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            #axs[2,1].set_ylim(min(-max(x0s)*1.1,-0.1*sigxs[0])*1e6, max(max(x0s)*1.1,0.1*sigxs[0])*1e6)
            axs[2,1].set_ylim(min(self.evolution.x_offset)*1e6*1.1, max(self.evolution.x_offset)*1e6*1.1)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
        
            return cax
        
        animation = FuncAnimation(fig, partial(frameFcn, file_list=files), frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots' + os.sep
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'sideview_x' + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename


    # Animate the horizontal angle density
    def animate_sideview_xp(self, evolution_folder):

        files = sorted(os.listdir(evolution_folder))
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam_init.set_qs(self.beam0_charge_sign*beam_init.qs())      #########
        
        max_sig_index = np.argmax(self.evolution.divergence_x)
        max_sig_beam = elegant_read_beam(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdzdxp0, zs0, xps0 = max_sig_beam.phase_space_density(max_sig_beam.zs, max_sig_beam.xps)
        dQdxp0, _ = max_sig_beam.projected_density(max_sig_beam.xps, bins=xps0)
        Is0, _ = max_sig_beam.current_profile(bins=zs0/SI.c)
        
        # get final beam
        beam_final = elegant_read_beam(evolution_folder + os.fsdecode(files[-1]))
        beam_final.set_qs(self.beam0_charge_sign*beam_final.qs())      #########
        dQdzdxp_final, zs_final, xps_final = beam_final.phase_space_density(beam_final.zs, beam_final.xps)
        dQdxp_final, _ = beam_final.projected_density(beam_final.xps, bins=xps0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        
        # prepare centroid arrays
        xp0s = []
        z0s = []
        Emeans = []
        sigzs = []
        ss = []
        emitns = []
        
        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
                
        # Frame function
        def frameFcn(i, file_list):
                    
            # get beam for this frame
            beam = elegant_read_beam(evolution_folder + os.fsdecode(file_list[i]))
            beam.set_qs(self.beam0_charge_sign*beam.qs())      #########
            
            # plot mean energy evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in interstage [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            #axs[0,0].set_ylim(0,beam_final.energy()*1.1e-9)
            axs[0,0].set_ylim(beam_init.energy()*0.9e-9, beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')

            # Plot lattice elements
            if i == 0:
                self.plot_element_lengths(ypos=0.0, height=0.15, ax=axs[0,0].twinx())
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_x()) # TODO: update to normalized amplitude
            sigzs.append(beam.bunch_length())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(min([min(emitns)*0.9e6, beam_final.norm_emittance_x()*0.8e6]), max([max(emitns)*1.1e6, emitns[0]*1.2e6]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, beam_final.bunch_length()*0.8e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdxp, zs, xps = beam.phase_space_density(beam.zs, beam.xps, hbins=zs0, vbins=xps0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, xps*1e6, -dQdzdxp, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse angle, $x\'$ [$\mathrm{\mu}$rad]")
            axs[1,0].set_title('Horizontal angle sideview')
            
            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot angle projection
            dQdxp, xps2 = beam.projected_density(beam.xps, bins=xps0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdxp, np.zeros(dQdxp.size)))*1e3, np.concatenate((xps2, np.flip(xps2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdxp*1e3, xps2*1e6, color=col1)
            axs[1,1].set_ylim([min(xps0)*1e6, max(xps0)*1e6])
            axs[1,1].set_xlim([0, max([max(-dQdxp), max(-dQdxp_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dx\'$ [nC/$\mathrm{\mu}$rad]")
            axs[1,1].set_ylabel(r"$x\'$ [$\mathrm{\mu}$rad]")
            
            # plot centroid evolution
            z0s.append(beam.z_offset())
            xp0s.append(beam.x_angle())
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(xp0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, xp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'$x\'$ offset [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(min([min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, max([max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            axs[2,1].set_ylim(min(self.evolution.x_angle)*1e6*1.1, max(self.evolution.x_angle)*1e6*1.1)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
        
            return cax
        
        animation = FuncAnimation(fig, partial(frameFcn, file_list=files), frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots' + os.sep
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'sideview_xp' + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename


    # Animate the horizontal phase space
    def animate_phasespace_x(self, evolution_folder):

        files = sorted(os.listdir(evolution_folder))
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam_init.set_qs(self.beam0_charge_sign*beam_init.qs())      #########
        
        max_sig_index = np.argmax(self.evolution.beam_size_x)
        max_sig_beam = elegant_read_beam(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdxdpx0, xs0, _ = max_sig_beam.phase_space_density(max_sig_beam.xs, max_sig_beam.pxs)
        
        max_sig_xp_index = np.argmax(self.evolution.divergence_x)
        max_sig_xp_beam = elegant_read_beam(evolution_folder + os.fsdecode(files[max_sig_xp_index]))
        _, _, pxs0 = max_sig_xp_beam.phase_space_density(max_sig_xp_beam.xs, max_sig_xp_beam.pxs)
        
        #dQdx0, _ = max_sig_beam.projected_density(max_sig_beam.xs, bins=xs0)
        #dQdpx0, _ = max_sig_xp_beam.projected_density(max_sig_xp_beam.pxs, bins=pxs0)
        dQdx0, _ = max_sig_xp_beam.projected_density(max_sig_xp_beam.xs, bins=xs0)
        dQdpx0, _ = max_sig_beam.projected_density(max_sig_beam.pxs, bins=pxs0)
        
        # get final beam
        beam_final = elegant_read_beam(evolution_folder + os.fsdecode(files[-1]))
        beam_final.set_qs(self.beam0_charge_sign*beam_final.qs())      #########
        dQdxdpx_final, xs_final, pxs_final = beam_final.phase_space_density(beam_final.xs, beam_final.pxs)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        dQdpx_final, _ = beam_final.projected_density(beam_final.pxs, bins=pxs0)
        
        # calculate limits
        pxlim = np.max(np.abs(pxs0))
        
        # prepare centroid arrays
        x0s = []
        xp0s = []
        sigxs = []
        sigxps = []
        ss = []
        emitns = []
        
        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i, file_list):
            # get beam for this frame
            beam = elegant_read_beam(evolution_folder + os.fsdecode(file_list[i]))
            beam.set_qs(self.beam0_charge_sign*beam.qs())      #########
            
            # plot emittance evolution
            ss.append(beam.location)
            emitns.append(beam.norm_emittance_x())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(emitns)*1e6, '-', color=col0)
            axs[0,0].plot(ss[-1], emitns[-1]*1e6, 'o', color=col1)
            axs[0,0].set_xlabel('Position in interstage [m]')
            axs[0,0].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(beam_init.norm_emittance_x()*0.5e6, np.max(self.evolution.norm_emittance_x)*1.2e6)
            axs[0,0].set_yscale('log')
            axs[0,0].yaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')

            # Plot lattice elements
            if i == 0:
                self.plot_element_lengths(ypos=0.0, height=0.15, ax=axs[0,0].twinx())
            
            # plot beam size and divergence evolution
            sigxs.append(beam.beam_size_x())
            sigxps.append(beam.divergence_x())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigxs)*1e6, np.array(sigxps)*1e3, '-', color=col0)
            axs[0,1].plot(sigxs[-1]*1e6, sigxps[-1]*1e3, 'o', color=col1)
            #axs[0,1].set_ylim(min([min(sigxps)*0.9e3, beam_final.divergence_x()*0.8e3]), max([max(sigxps)*1.1e3, sigxs[0]*1.2e3]))
            axs[0,1].set_ylim(np.min(self.evolution.divergence_x)*0.8e3, max_sig_xp_beam.divergence_x()*1.2e3)
            #axs[0,1].set_xlim(min([min(sigxs)*0.9e6, beam_final.beam_size_x()*0.8e6]), max([max(sigxs)*1.1e6, sigxs[0]*1.2e6]))
            axs[0,1].set_xlim(beam_init.beam_size_x()*0.8e6, max_sig_beam.beam_size_x()*1.2e6)
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Beam size [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Divergence [mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdxdpx, xs, pxs = beam.phase_space_density(beam.xs, beam.pxs, hbins=xs0, vbins=pxs0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(xs*1e6, pxs*1e-6*SI.c/SI.e, -dQdxdpx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel("Momentum, $p_x$ [MeV/c]")
            axs[1,0].set_title('Horizontal phase space')
            axs[1,0].set_ylim([-pxlim*1e-6*SI.c/SI.e, pxlim*1e-6*SI.c/SI.e])
            
            # plot position projection
            af = 0.15
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((xs2, np.flip(xs2)))*1e6, -np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, alpha=af, color=col1)
            axs[2,0].plot(xs2*1e6, -dQdx*1e3, color=col1)
            axs[2,0].set_xlim([min(xs0)*1e6, max(xs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-dQdx0), max(-dQdx_final)])*1.2e3])
            axs[2,0].set_xlabel(r'Transverse position, $x$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$dQ/dx$ [nC/$\mathrm{\mu}$m]')
            
            # plot angular projection
            dQdpx, pxs2 = beam.projected_density(beam.pxs, bins=pxs0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdpx, np.zeros(dQdpx.size)))*1e9/(1e-6*SI.c/SI.e), np.concatenate((pxs2, np.flip(pxs2)))*1e-6*SI.c/SI.e, alpha=af, color=col1)
            axs[1,1].plot(-dQdpx*1e9/(1e-6*SI.c/SI.e), pxs2*1e-6*SI.c/SI.e, color=col1)
            axs[1,1].set_ylim([-pxlim*1e-6*SI.c/SI.e, pxlim*1e-6*SI.c/SI.e])
            axs[1,1].set_xlim([0, max([max(-dQdpx0), max(-dQdpx_final)])*1e9/(1e-6*SI.c/SI.e)*1.2])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dp_x$ [nC c/MeV]")
            axs[1,1].set_ylabel("Momentum, $p_x$ [MeV/c]")
            
            # plot centroid evolution
            x0s.append(beam.x_offset())
            xp0s.append(beam.x_angle())
            axs[2,1].cla()
            axs[2,1].plot(np.array(x0s)*1e6, np.array(xp0s)*1e6, '-', color=col0)
            axs[2,1].plot(x0s[-1]*1e6, xp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'Centroid offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'Centroid angle [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(min(self.evolution.x_offset)*1e6*1.1, max(self.evolution.x_offset)*1e6*1.1)
            axs[2,1].set_ylim(min(self.evolution.x_angle)*1e6*1.1, max(self.evolution.x_angle)*1e6*1.1)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
            
            return cax
        
        # make all frames
        animation = FuncAnimation(fig, partial(frameFcn, file_list=files), frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'phasespace_x' + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename


    # Animate the longitudinal phase space
    def animate_lps(self, evolution_folder, rel_energy_window=0.06):

        files = sorted(os.listdir(evolution_folder))
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam_init.set_qs(self.beam0_charge_sign*beam_init.qs())      #########
        
        max_sig_E_index = np.argmax(self.evolution.rel_energy_spread)
        max_sig_E_beam = elegant_read_beam(evolution_folder + os.fsdecode(files[max_sig_E_index]))
        _, _, Es0 = max_sig_E_beam.phase_space_density(max_sig_E_beam.zs, max_sig_E_beam.Es)
        
        dQdzdE0, zs0, Es0 = max_sig_E_beam.density_lps()
        Is0, ts_ = max_sig_E_beam.current_profile(bins=zs0/SI.c)
        dQdE0, Es_ = max_sig_E_beam.energy_spectrum(bins=Es0)

        # get final beam
        beam_final = elegant_read_beam(evolution_folder + os.fsdecode(files[-1]))
        beam_final.set_qs(self.beam0_charge_sign*beam_final.qs())      #########
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        dQdE_final, _ = beam_final.energy_spectrum(bins=Es0)

        # nominal energies
        Es_nom = self.nom_energy

        # calculate limits
        E_lim_max = np.max(np.abs(Es0))
        E_lim_min = np.min(np.abs(Es0))
        
        # prepare centroid arrays
        z0s = []
        deltas = []
        sigzs = []
        sigdeltas = []
        ss = []
        Emeans = []

        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i, file_list):

            # get beam for this frame
            beam = elegant_read_beam(evolution_folder + os.fsdecode(file_list[i]))
            beam.set_qs(self.beam0_charge_sign*beam.qs())      #########
            
            # plot mean energy evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in interstage [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(beam_init.energy()*0.9e-9, beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')

            # Plot lattice elements
            if i == 0:
                self.plot_element_lengths(ypos=0.0, height=0.15, ax=axs[0,0].twinx())
            
            # plot energy spread and bunch length evolution
            sigzs.append(beam.bunch_length())
            sigdeltas.append(beam.rel_energy_spread())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(sigdeltas)*1e2, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, sigdeltas[-1]*1e2, 'o', color=col1)
            axs[0,1].set_ylim(min([min(sigdeltas)*0.8e2, 1e-1]), max([max(sigdeltas)*1.2e2, 10]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, sigzs[0]*0.7e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.3e6]))
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Energy spread [%]')
            axs[0,1].set_yscale('log')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            
            # plot LPS
            axs[1,0].cla()
            #Ebins = Es_nom*np.linspace(1-rel_energy_window, 1+rel_energy_window, 2*Es0.size)
            Ebins = np.linspace(E_lim_min, E_lim_max, 2*Es0.size)
            dQdzdE, zs, Es = beam.phase_space_density(beam.zs, beam.Es, hbins=zs0, vbins=Ebins)
            cax = axs[1,0].pcolor(zs*1e6, Es/1e9, -dQdzdE*1e15, cmap=CONFIG.default_cmap, shading='auto', clim=[0, abs(dQdzdE).max()*1e15])
            axs[1,0].set_ylabel('Energy [GeV]')
            axs[1,0].set_title('Longitudinal phase space')
            axs[1,0].set_ylim([E_lim_min/1e9, E_lim_max/1e9])
            
            # plot current profile
            axs[2,0].cla()
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot energy spectrum
            dQdE, Es2 = beam.energy_spectrum(bins=Ebins)
            deltas2 = Es2/Es_nom-1
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdE, np.zeros(dQdE.size)))*1e18, np.concatenate((deltas2, np.flip(deltas2)))*100, alpha=af, color=col1)
            axs[1,1].plot(-dQdE*1e18, deltas2*100)
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].set_ylabel('Rel. energy [%]')
            #axs[1,1].set_ylim([-rel_energy_window*100, rel_energy_window*100])
            axs[1,1].set_ylim([(E_lim_min/Es_nom-1)*100, (E_lim_max/Es_nom-1)*100])
            axs[1,1].set_xlim([0, max(-dQdE)*1.1e18])
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel('dQ/dE [nC/GeV]')
            
            # plot E and z centroid evolution
            z0s.append(beam.z_offset())
            deltas.append(beam.energy()/Es_nom-1)
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(deltas)*100, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, deltas[-1]*100, 'o', color=col1)
            #axs[2,1].set_ylim([-rel_energy_window*0.5e2, rel_energy_window*0.5e2])
            axs[2,1].set_ylim([(E_lim_min/Es_nom-1)*100, (E_lim_max/Es_nom-1)*100])
            axs[2,1].set_xlim([(z0s[0]-sigzs[0]/2)*1e6, (z0s[0]+sigzs[0]/2)*1e6])
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel('Energy offset [%]')
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')

            return cax

        # make all frames
        animation = FuncAnimation(fig, partial(frameFcn, file_list=files), frames=range(len(files)), repeat=False, interval=100)
        

        # save the animation as a GIF
        plot_path = self.run_path + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'lps_shot' + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename

    
    # Apply waterfall function to all beam files
    def __waterfall_fcn(self, fcns, edges, evolution_folder, args=None, shot=0):
        
        # find number of beam outputs to plot
        files = sorted(os.listdir(evolution_folder))
        num_outputs = len(files)
        beam_init = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam_init.set_qs(self.beam0_charge_sign*beam_init.qs())                                                        ############################
        
        # declare data structure
        bins = [None] * len(fcns)
        waterfalls = [None] * len(fcns)
        for j in range(len(fcns)):
            waterfalls[j] = np.empty((len(edges[j])-1, num_outputs))
        
        locations = np.empty(num_outputs)
        #locations = [None] * len(fcns)
        
        # go through files
        for index in range(num_outputs):
            # load phase space
            #beam = self.get_beam(index=index, shot=shot)
            beam = elegant_read_beam(evolution_folder + os.fsdecode(files[index]))
            beam.set_qs(self.beam0_charge_sign*beam.qs())                                                        # Patch for ELEGANT not setting the correct charge sign.
            
            # find trackable number
            locations[index] = beam.location
            
            # get all waterfalls (apply argument if it exists)
            for j in range(len(fcns)):
                if args[j] is None:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, bins=edges[j])
                else:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, args[j][index], bins=edges[j])
                
        return waterfalls, locations, bins

    
    # Waterfall plots
    def plot_waterfalls(self, evolution_folder, shot=None, save_fig=False):
        
        if self.save_evolution is False:
            raise Exception('ELEGANT beam evolution files have not been saved.')
            
        # select shot
        if shot is None:
            if hasattr(self, 'shot') and self.shot is not None:
                shot = self.shot
            else:
                shot = 0
        
        # calculate values
        #beam0 = self.get_beam(0,shot=shot)
        files = sorted(os.listdir(evolution_folder))
        beam0 = elegant_read_beam(evolution_folder + os.fsdecode(files[0]))
        beam0.set_qs(self.beam0_charge_sign*beam0.qs())                                                        ############################
        num_bins = int(np.sqrt(len(beam0)*2))
        nsig = 4
        
        tedges = (beam0.z_offset(clean=True) + nsig*beam0.bunch_length(clean=True)*np.linspace(-1, 1, num_bins)) / SI.c
        #deltaedges = np.linspace(-0.1, 0.1, num_bins)
        deltaedges = (np.max(self.evolution.rel_energy_spread)*nsig + abs(beam0.energy()-self.nom_energy)/self.nom_energy) * np.linspace(-1, 1, num_bins)
        xedges = (nsig*np.max(self.evolution.beam_size_x) + abs(beam0.x_offset()))*np.linspace(-1, 1, num_bins)
        yedges = (nsig*np.max(self.evolution.beam_size_y) + abs(beam0.y_offset()))*np.linspace(-1, 1, num_bins)
        E0s = self.nom_energy*np.ones(len(files))
        xpedges = (nsig*np.max(np.abs(self.evolution.divergence_x)) + abs(beam0.x_angle()))*np.linspace(-1, 1, num_bins)
        #ypedges = (nsig*np.max(np.abs(self.evolution.y_angle)) + abs(beam0.y_angle()))*np.linspace(-1, 1, num_bins)
        
        waterfalls, locations, bins = self.__waterfall_fcn([Beam.current_profile, Beam.rel_energy_spectrum, Beam.transverse_profile_x, Beam.transverse_profile_y, Beam.transverse_profile_xp], [tedges, deltaedges, xedges, yedges, xpedges], evolution_folder, [None, E0s, None, None, None], shot)
        
        # prepare figure
        fig, axs = plt.subplots(5,1)
        fig.set_figwidth(8)
        fig.set_figheight(2.8*5)
        
        # current profile
        Is = waterfalls[0]
        ts = bins[0]
        c0 = axs[0].pcolor(locations, ts*SI.c*1e6, -Is/1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_ylabel(r'Longitudinal position [$\mathrm{\mu}$m]')
        cbar0.ax.set_ylabel('Beam current [kA]')
        #axs[0].set_title('Shot ' + str(shot+1))
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(locations, deltas*1e2, -dQddeltas*1e7, cmap=CONFIG.default_cmap, shading='auto')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_ylabel('Energy offset [%]')
        cbar1.ax.set_ylabel('Spectral density [nC/%]')
        
        densityX = waterfalls[2]
        xs = bins[2]
        c2 = axs[2].pcolor(locations, xs*1e6, -densityX*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar2 = fig.colorbar(c2, ax=axs[2])
        axs[2].set_ylabel(r'Horizontal position [$\mathrm{\mu}$m]')
        cbar2.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')
        
        densityY = waterfalls[3]
        ys = bins[3]
        c3 = axs[3].pcolor(locations, ys*1e6, -densityY*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar3 = fig.colorbar(c3, ax=axs[3])
        axs[3].set_ylabel(r'Vertical position [$\mathrm{\mu}$m]')
        cbar3.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')

        densityXp = waterfalls[4]
        xps = bins[4]
        c4 = axs[4].pcolor(locations, xps*1e6, -densityXp*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar4 = fig.colorbar(c4, ax=axs[4])
        axs[4].set_ylabel(r'Horizontal angle [$\mathrm{\mu}$rad]')
        cbar4.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$rad]')

        axs[4].set_xlabel('Location along interstage [m]')

        #densityYp = waterfalls[5]
        #yps = bins[5]
        #c5 = axs[5].pcolor(locations, yps*1e6, -densityYp*1e3, cmap=CONFIG.default_cmap, shading='auto')
        #cbar5 = fig.colorbar(c5, ax=axs[5])
        #axs[5].set_ylabel(r'Vertical angle [$\mathrm{\mu}$rad]')
        #cbar5.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$rad]')
        
        plt.show()
        if save_fig:
            plot_path = self.run_path + 'plots' + os.sep
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'waterfalls' + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)

    
    def plot_rectangle(self, xpos, length, ypos=0.25, height=0.5, color='blue', ax=None):
        if ax is None:
            ax = plt.gca()
    
        rectangle = Rectangle((xpos, ypos), length, height, color=color, alpha=0.5)
        ax.add_patch(rectangle)
        return rectangle

    
    def plot_element_lengths(self, ypos=0.25, height=0.5, ax=None):
        ls = self.element_lengths

        # Calculate cumulative lengths
        cumulative_lengths = np.cumsum(ls)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(20)
            fig.set_figheight(1)
        self.plot_rectangle(xpos=cumulative_lengths[0], length=ls[1], ypos=ypos, height=height, color='red', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[2], length=ls[3], ypos=ypos, height=height, ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[4], length=ls[5], ypos=ypos, height=height, color='orange', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[6], length=ls[7], ypos=ypos, height=height, color='orange', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[8], length=ls[9], ypos=ypos, height=height, color='purple', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[10], length=ls[11], ypos=ypos, height=height, color='orange', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[12], length=ls[13], ypos=ypos, height=height, color='orange', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[14], length=ls[15], ypos=ypos, height=height, ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[16], length=ls[17], ypos=ypos, height=height, color='red', ax=ax)
        self.plot_rectangle(xpos=cumulative_lengths[17], length=ls[18], ypos=ypos, height=height, color='white', ax=ax)
        ax.set_xlim(0, cumulative_lengths[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel('s [m]')
        ax.set_yticks([])
        ax.set_yticklabels([])
        