# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import os, uuid, subprocess, csv
import numpy as np
from abel.CONFIG import CONFIG
from abel.classes.event import Event


def guineapig_run(inputfile, beam1, beam2, tmpfolder=None):
    """
    Run a GUINEA-PIG beam–beam interaction simulation between two beams.

    This function executes a GUINEA-PIG simulation using the provided GUINEA-PIG 
    input file and two colliding ``Beam`` objects. It prepares the temporary 
    beam input files, runs the external GUINEA-PIG binary, parses the resulting 
    output file (``.ref``-file), and returns an ``Event`` object containing the 
    incoming and outgoing beams, as well as luminosity values and other 
    interaction results.
    

    Parameters
    ----------
    inputfile : str
        Path to the GUINEA-PIG accelerator file. This file defines the beam–beam 
        interaction parameters such as crossing angle, magnetic fields, and 
        bunch spacing.

    beam1 : ``Beam``
        First input beam. Used to generate the GUINEA-PIG input file 
        ``inputbeam1.ini``.

    beam2 : ``Beam``
        Second input beam. Analogous to ``beam1``, it is used to generate 
        ``inputbeam2.ini`` for use by GUINEA-PIG.

    tmpfolder : str, optional
        Path to a temporary working directory to store intermediate files (e.g. 
        GUINEA-PIG outputs). If ``None``, a unique folder is created in 
        ``CONFIG.temp_path``. Defaults to ``None``.


    Returns
    -------
    event : ``Event``
        Object representing the simulated beam–beam interaction, 
        containing:

        - ``beam_in_1``, ``beam_in_2`` : Input beams.
        - ``beam_out_1``, ``beam_out_2`` : Outgoing beams after collision.
        - ``luminosity_geom`` : [m^-2 per bunch crossing] Geometric luminosity.
        - ``luminosity_full`` : [m^-2 per bunch crossing] Full luminosity.
        - ``luminosity_peak`` : [m^-2 per bunch crossing] Peak luminosity.
        - ``upsilon_max`` : Maximum beamstrahlung parameter.
        - ``num_pairs`` : Number of coherent pairs (per macroparticle?).
        - ``num_photon1``, ``num_photon2`` : Number of emitted beamstrahlung photons.
        - ``energy_loss1``, ``energy_loss2`` : [eV] Energy loss per particle.


    References
    ----------
    .. [1] D. Schulte, "Study of Electromagnetic and Hadronic Background in the Interaction Region of the TESLA Collider", University of Hamburg (1997)
    """
    
    # make temporary output file
    if tmpfolder is None:
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
        os.mkdir(tmpfolder)
    outputfile = "output.ref"
    
    # make temporary beam files
    beamfile1 = "inputbeam1.ini"
    beamfile2 = "inputbeam2.ini"
    beamfile1_fullpath = tmpfolder + "/" + beamfile1
    beamfile2_fullpath = tmpfolder + "/" + beamfile2
    guineapig_write_beam(beam1, beamfile1_fullpath)
    guineapig_write_beam(beam2, beamfile2_fullpath)

    # run GUINEA-PIG
    cmd = 'cd ' + tmpfolder + '; ' + os.path.join(CONFIG.guineapig_path, 'guinea') + ' default default ' + outputfile + ' --el_file=' + beamfile1 + ' --pos_file=' + beamfile2 + ' --acc_file=' + inputfile
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    
    # parse outputs
    outputfile_fullpath = tmpfolder + '/' + outputfile
    lumi_ee_full, lumi_ee_peak, lumi_ee_geom, upsilon_max, num_pairs, num_photon1, num_photon2, energy_loss1, energy_loss2 = guineapig_read_output(outputfile_fullpath)

    # extract outgoing beams
    beamfile1_out = tmpfolder + "/beam1.dat"
    beamfile2_out = tmpfolder + "/beam2.dat"
    beam_out1 = guineapig_read_beam(beamfile1_out, Q=beam1.charge(), beta_x=beam1.beta_x(), beta_y=beam1.beta_y(), z_mean=beam1.z_offset())
    beam_out2 = guineapig_read_beam(beamfile2_out, Q=beam2.charge(), beta_x=beam2.beta_x(), beta_y=beam2.beta_y(), z_mean=beam2.z_offset())
    
    # remove temporary files and folders
    os.remove(outputfile_fullpath)
    os.remove(beamfile1_fullpath)
    os.remove(beamfile2_fullpath)
    
    # make event object
    event = Event(beam1, beam2, beam_out1, beam_out2)
    event.luminosity_geom = lumi_ee_geom
    event.luminosity_full = lumi_ee_full
    event.luminosity_peak = lumi_ee_peak
    event.upsilon_max = upsilon_max
    event.num_pairs = num_pairs
    event.num_photon1 = num_photon1
    event.num_photon2 = num_photon2
    event.energy_loss1 = energy_loss1
    event.energy_loss2 = energy_loss2
    
    return event
    

# ==================================================
def guineapig_write_beam(beam, filename, beta_x=None, beta_y=None):
    """
    Write an ABEL ``Beam``object to a GUINEA-PIG input beam file.

    Parameters
    ----------
    beam : ``Beam``
        ABEL ``Beam``object to be written to file

    filename : str
        Path to the GUINEA-PIG input beam file (a ``.ini``-file). 

    beta_x : [m] float, optional
        The beta function in x of the beam. If ``None``, it is calculated 
        directly from ``beam``.  Defaults to ``None``.

    beta_y : [m] float, optional
        The beta function in y of the beam. If ``None``, it is calculated 
        directly from ``beam``.  Defaults to ``None``.

    Returns
    -------
    None
        The function writes data to disk and does not return a value.
    """

    # extract beta function (for normalization)
    if beta_x is None:
        beta_x = beam.beta_x()
    if beta_y is None:
        beta_y = beam.beta_y()
        
    # write beam phasespace to CSV
    Es = beam.Es()
    zs = beam.z_offset()-beam.zs() # opposite sign and centered around the middle
    xs_ipslice_norm = beam.xs()/beta_x # position of each particle when passing through z=0, normalized by beta
    ys_ipslice_norm = beam.ys()/beta_y
    xps_norm = beam.xps()*beta_x # particle angle, normalized by 1/beta
    yps_norm = beam.yps()*beta_y
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f, delimiter=' ')
        for i in range(int(len(beam))):
            csvwriter.writerow([Es[i]/1e9, xps_norm[i]*1e6, yps_norm[i]*1e6, zs[i]*1e6, xs_ipslice_norm[i]*1e6, ys_ipslice_norm[i]*1e6])


# ==================================================
def guineapig_read_beam(filename, Q, beta_x, beta_y, z_mean=0.0):
    """
    Extract data from a GUINEA-PIG output beam file and convert it into a 
    ``Beam`` object.

    
    Parameters
    ----------
    filename : str
        Path to the GUINEA-PIG output beam file (usually ``beam1.dat`` and 
        ``beam2.dat``). Containing the particles of an output beam.

    Q : [C] float
        Total beam charge.

    beta_x : [m] float
        The beta function in x of the beam.

    beta_y : [m] float
        The beta function in y of the beam.

    z_mean : [m] float, optional
        The mean z-position of the beam. Defaults to 0.0.


    Returns
    -------
    beam : ``Beam``
    """
    
    # declare variables
    Es = []
    zs = []
    xps = []
    yps = []
    xs = []
    ys = []
    
    # perform CSV extraction
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader: # TODO: re-write this to extract data more efficiently. 
            
            E = float(row[0].strip())*1e9
            Es.append(E)
            
            xp = float(row[1].strip())/beta_x*1e-6
            xps.append(xp)
            
            yp = float(row[2].strip())/beta_y*1e-6
            yps.append(yp)
            
            z = z_mean - float(row[3].strip())*1e-6
            zs.append(z)

            x = float(row[4].strip())*beta_x*1e-6
            xs.append(x)

            y = float(row[5].strip())*beta_y*1e-6
            ys.append(y)

    # make beam
    from abel.classes.beam import Beam
    beam = Beam()
    beam.set_phase_space(Q=Q, Es=np.array(Es), xps=np.array(xps), yps=np.array(yps), zs=np.array(zs), xs=np.array(xs), ys=np.array(ys))
    
    return beam
    

# ==================================================
def guineapig_read_output(outputfile, verbose=False):
    """
    Parse and extract beam–beam interaction results from a GUINEA-PIG output 
    file.

    This function reads the output file produced by GUINEA-PIG (typically named
    ``output.ref``) and extracts key diagnostic quantities such as luminosities,
    beamstrahlung parameters, and particle production counts.
    

    Parameters
    ----------
    outputfile : str
        Path to the GUINEA-PIG output file (usually ``output.ref``). The file is
        assumed to be generated by a standard GUINEA-PIG run and contain the
        diagnostic summary of the beam–beam interaction.

    verbose : bool, optional
        If ``True``, print each line of the file to the console while parsing. 
        Defaults to ``False``.


    Returns
    -------
    lumi_ee_full : [m^-2 per bunch crossing] float
        Full luminosity per bunch crossing, including all beam–beam effects.

    lumi_ee_peak : [m^-2 per bunch crossing] float
        Peak luminosity per bunch crossing, typically corresponding to the
        high-energy core of the beam distribution.

    lumi_ee_geom : [m^-2 per bunch crossing] float
        Geometric luminosity, computed assuming ideal (non-disrupted) beams.

    upsilon_max : float
        Maximum beamstrahlung parameter (Υ) that occured duting the interaction, 
        describing the strength of the electromagnetic fields during the 
        collision.

    num_pairs : float
        Number of incoherent e+e- pairs produced via beam–beam interactions.

    num_photon1 : float
        Number of beamstrahlung photons emitted per macroparticle in beam 1.

    num_photon2 : float
        Number of beamstrahlung photons emitted per macroparticle in beam 2.

    energy_loss1 : [eV] float
        Mean energy loss of beam particles in beam 1 due to beam–beam 
        interactions.

    energy_loss2 : [eV] float
        Mean energy loss of beam particles in beam 1 due to beam–beam 
        interactions.


    References
    ----------
    .. [1] D. Schulte, "Study of Electromagnetic and Hadronic Background in the Interaction Region of the TESLA Collider", University of Hamburg (1997)
    """

    lumi_ee_full = None
    lumi_ee_peak = None
    lumi_ee_geom = None
    upsilon_max = None
    num_pairs = None
    num_photon1 = None
    num_photon2 = None
    energy_loss2 = None
    energy_loss2 = None
    
    # string to search in file
    with open(outputfile, 'r') as fp:
        lines = fp.readlines()
        for row in lines:

            # print all if verbose
            if verbose:
                print(row.split('\n')[0])

            # extract parameters from file
            if row.find('lumi_fine ') != -1:
                lumi_ee_geom = float(row.split(" ")[2]) # [m^-2 per crossing]
            if row.find('lumi_ee ') != -1:
                lumi_ee_full = float(row.split(" ")[2]) # [m^-2 per crossing]
            if row.find('lumi_ee_high ') != -1:
                lumi_ee_peak = float(row.split(" ")[2]) # [m^-2 per crossing]
            if row.find('upsmax= ') != -1:
                upsilon_max = float(row.split(" ")[1])
            if row.find('n_pairs =') != -1:
                num_pairs = float(row.split(" : ")[1].split(" ")[2]) # TODO: this is used to return the number of coherent pairs in Event.num_coherent_pairs(), but this is actually the number of incoherent pairs.
            if row.find('final number of phot. per tracked macropart.1') != -1:
                num_photon1 = float(row.split(" : ")[1].strip(' '))
            if row.find('final number of phot. per tracked macropart.2') != -1:
                num_photon2 = float(row.split(" : ")[1].strip(' '))
            if row.find('de1=') != -1:
                energy_loss1 = float(row.split("=")[1].split(";")[0].strip(' '))*1e9
            if row.find('de2=') != -1:
                energy_loss2 = float(row.split("=")[1].split(";")[0].strip(' '))*1e9
    
    return lumi_ee_full, lumi_ee_peak, lumi_ee_geom, upsilon_max, num_pairs, num_photon1, num_photon2, energy_loss1, energy_loss2
