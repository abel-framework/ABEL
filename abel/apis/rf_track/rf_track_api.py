import scipy.constants as SI
import numpy as np


# ==================================================
def abel_beam2rft_beam(beam, homogen_beam_charge=True):
    """
    Converts an ABEL ``Beam`` object to a RF-Track ``Bunch6dT`` object.

    Parameters
    ----------
    beam : ABEL ``Beam`` object
        The beam to be converted.

    homogen_beam_charge : bool, optional
        Flag for indicating the whether the macroparticles of ``beam`` all have 
        the same charges. Defaults to ``True``, which allows for using a faster 
        version of the ``Bunch6dT`` constructor.


    Returns
    ----------
    beam_rft : RF-Track ``Bunch6dT`` object
    """

    from RF_Track import Bunch6dT

    xs_abel = beam.xs()    # [m]
    pxs_abel = beam.pxs()  # [kg m/s]
    ys_abel = beam.ys()
    pys_abel = beam.pys()
    zs_abel = beam.zs()
    pzs_abel = beam.pzs()
    qs_abel = beam.qs()
    weightings_abel = beam.weightings()

    # Hack for setting the weight for macroparticles with 0 charge. This hack is used in ion_motion_wakefield_perturbation.py to add "ghost particles" in order to enlarge the box for calculating the beam fields using RF-Track.
    zero_mask = qs_abel == 0
    if sum(zero_mask) != 0:
        weightings_abel[zero_mask] = weightings_abel[~zero_mask][0]

    particle_mass = beam.particle_mass*SI.c**2/SI.e/1e6  # [MeV/c^2]
    
    if not homogen_beam_charge:   
        # Convert the phase space to RFT units and in the format [ X Px Y Py Z Pz MASS Q N ]
        ms_abel = particle_mass * np.ones(len(beam))  # [MeV/c^2] single particle masses.
        phase_space_rft = np.column_stack((xs_abel*1e3, pxs_abel*SI.c/SI.e/1e6, 
                                        ys_abel*1e3, pys_abel*SI.c/SI.e/1e6, 
                                        zs_abel*1e3, pzs_abel*SI.c/SI.e/1e6, 
                                        ms_abel, qs_abel/SI.e/weightings_abel, 
                                        weightings_abel))

        # Construct a RFT beam
        beam_rft = Bunch6dT(phase_space_rft)

    else:
        # Convert the phase space to RFT units and in the format [ X Px Y Py Z Pz ] 
        phase_space_rft = np.column_stack((xs_abel*1e3, pxs_abel*SI.c/SI.e/1e6, 
                                        ys_abel*1e3, pys_abel*SI.c/SI.e/1e6, 
                                        zs_abel*1e3, pzs_abel*SI.c/SI.e/1e6))
        
        # Construct a RFT beam using Bunch6dT(mass, population, charge, [ X Px Y Py Z Pz ] )
        single_particle_charge = qs_abel[0]/SI.e/weightings_abel[0]  # Charge of a single physical particle [e].
        beam_rft = Bunch6dT(particle_mass, beam.population(), single_particle_charge, phase_space_rft)
    
    return beam_rft


# ==================================================
def rft_beam2abel_beam(beam_rft):
    """
    Converts a RF-Track ``Bunch6dT`` object to an ABEL ``Beam`` object.
    """
    
    import warnings
    from abel.classes.beam import Beam

    phase_space_rft = beam_rft.get_phase_space('%X %Px %Y %Py %Z %Pz %Q %N %m', 'good')
    xs_rft = phase_space_rft[:,0]  # [mm]
    ys_rft = phase_space_rft[:,2]
    zs_rft = phase_space_rft[:,4]
    pxs_rft = phase_space_rft[:,1]  # [MeV/c]
    pys_rft = phase_space_rft[:,3]
    pzs_rft = phase_space_rft[:,5]

    weightings = phase_space_rft[:,7]
    tot_charge = np.sum(phase_space_rft[:,6] * weightings)*SI.e
    
    particle_masses = phase_space_rft[:,8]  # [MeV/c^2]
    if np.abs(particle_masses.min()/particle_masses.max() - 1) > 1e-5:
        warnings.warn('The RF-Track beam contains different single particle masses.')
    
    particle_mass = particle_masses[0]*1e6*SI.e/SI.c**2
    
    # Create an empty beam
    beam = Beam()

    # Set the phase space
    beam.set_phase_space(xs=xs_rft/1e3, ys=ys_rft/1e3, zs=zs_rft/1e3, 
                         pxs=pxs_rft*1e6*SI.e/SI.c, pys=pys_rft*1e6*SI.e/SI.c, 
                         pzs=pzs_rft*1e6*SI.e/SI.c, 
                         Q=tot_charge, weightings=weightings, 
                         particle_mass=particle_mass)
    return beam

    
# ==================================================
def calc_sc_fields_obj(abel_beam, num_x_cells, num_y_cells, num_z_cells=None, num_t_bins=1):

    from RF_Track import SpaceCharge_Field
    
    if num_z_cells is None:
        num_z_cells = round(np.sqrt(len(abel_beam))/2)

    # Convert ABEL beam to RF-Track beam
    beam_rft = abel_beam2rft_beam(abel_beam, homogen_beam_charge=True)  # Add a flag and edit this so that it is also compatible with Wake-T ParticleBunch
        
    # Set the solver resolution and calculate fields
    sc_fields_obj = SpaceCharge_Field(beam_rft, num_x_cells, num_y_cells, num_z_cells, num_t_bins)  # num_x_cells, num_y_cells, num_z_cells, number of velocity slices
    
    return sc_fields_obj

    
# ==================================================
def rft_beam_fields(abel_beam, num_x_cells, num_y_cells, num_z_cells=None, num_t_bins=1, sort_zs=False):
    
    # Set the solver resolution and calculate fields
    sc_fields_obj = calc_sc_fields_obj(abel_beam, num_x_cells, num_y_cells, num_z_cells, num_t_bins)

    # Extract beam coordinates
    #phase_space_rft = beam_rft.get_phase_space('%X %Y %Z', 'good')
    #xs = phase_space_rft[:,0]  # [mm]
    #ys = phase_space_rft[:,1]
    #zs = phase_space_rft[:,2]
    xs = abel_beam.xs()*1e3  # [mm]
    ys = abel_beam.ys()*1e3  # [mm]
    zs = abel_beam.zs()*1e3  # [mm]

    # Sort the arrays based on zs.
    if sort_zs:        
        indices = np.argsort(zs)
        zs_sorted = zs[indices]
        xs_sorted = xs[indices]
        ys_sorted = ys[indices]
    else:
        zs_sorted = zs
        xs_sorted = xs
        ys_sorted = ys

    # Evaluate the electric field at the sorted coordinates so that the fields are sorted according to zs
    E_fields_beam, B_fields_beam = sc_fields_obj.get_field(xs_sorted, ys_sorted, zs_sorted, np.zeros(len(xs_sorted)))

    xs_sorted = xs_sorted/1e3  # [m]
    ys_sorted = ys_sorted/1e3  # [m]
    zs_sorted = zs_sorted/1e3  # [m]

    return E_fields_beam, B_fields_beam, xs_sorted, ys_sorted, zs_sorted


# ==================================================
def wake_t_bunch2rft_beam(wake_t_bunch, homogen_beam_charge=True):
    """
    Converts a Wake-T ``ParticleBunch`` to a RF-Track ``Bunch6dT`` object.

    Parameters
    ----------
    wake_t_bunch : Wake-T ``ParticleBunch``
        The beam to be converted.

    homogen_beam_charge : bool, optional
        Flag for indicating the whether the macroparticles of ``wake_t_bunch`` 
        all have the same charges. Defaults to ``True``, which allows for using 
        a faster version of the ``Bunch6dT`` constructor.


    Returns
    ----------
    beam_rft : RF-Track ``Bunch6dT`` object
    """

    from RF_Track import Bunch6dT

    xs_wt = wake_t_bunch.x    # [m]
    pxs_wt = wake_t_bunch.px  # [kg m/s / (m c)] p/
    ys_wt = wake_t_bunch.y
    pys_wt = wake_t_bunch.py
    zs_wt = wake_t_bunch.xi
    pzs_wt = wake_t_bunch.pz
    weightings_wt = wake_t_bunch.w

    particle_mass = wake_t_bunch.m_species * SI.c**2 / SI.e / 1e6  # [MeV/c^2], also the same as the conversion factor from Wake-T momenta to RF-Track momenta.
    particle_charge = wake_t_bunch.q_species  # [C] single particle charge
    
    if not homogen_beam_charge:   
        # Convert the phase space to RFT units and in the format [ X Px Y Py Z Pz MASS Q N ]
        ms = particle_mass * np.ones(len(xs_wt))  # [MeV/c^2] single particle masses.
        qs = particle_charge/SI.e * np.ones(len(xs_wt))  # [e] single particle charges.
        
        phase_space_rft = np.column_stack((xs_wt*1e3, pxs_wt*particle_mass, 
                                        ys_wt*1e3, pys_wt*particle_mass, 
                                        zs_wt*1e3, pzs_wt*particle_mass, 
                                        ms, qs, 
                                        weightings_wt))

        # Construct a RFT beam
        beam_rft = Bunch6dT(phase_space_rft)

    else:
        # Convert the phase space to RFT units and in the format [ X Px Y Py Z Pz ] 
        phase_space_rft = np.column_stack((xs_wt*1e3, pxs_wt*particle_mass, 
                                        ys_wt*1e3, pys_wt*particle_mass, 
                                        zs_wt*1e3, pzs_wt*particle_mass))
        
        # Construct a RFT beam using Bunch6dT(mass, population, charge, [ X Px Y Py Z Pz ] )
        beam_rft = Bunch6dT(particle_mass, len(xs_wt), particle_charge/SI.e, phase_space_rft)
    
    return beam_rft


# # ==================================================
# def rft_beam2wake_t_bunch(beam_rft):
#     """
#     Converts a RF-Track ``Bunch6dT`` object to a Wake-T ``ParticleBunch``.
#     """

#     return
