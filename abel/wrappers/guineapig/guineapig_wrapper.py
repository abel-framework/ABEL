# Copyright 2022-, The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

import os, uuid, subprocess, csv
import numpy as np
from abel.CONFIG import CONFIG
from abel.classes.event import Event

def guineapig_run(inputfile, beam1, beam2, tmpfolder=None):
    
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
    

def guineapig_write_beam(beam, filename, beta_x=None, beta_y=None):

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


def guineapig_read_beam(filename, Q, beta_x, beta_y, z_mean=0):
    
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
        for row in reader:
            
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
    

def guineapig_read_output(outputfile, verbose=False):

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
                num_pairs = float(row.split(" : ")[1].split(" ")[2])
            if row.find('final number of phot. per tracked macropart.1') != -1:
                num_photon1 = float(row.split(" : ")[1].strip(' '))
            if row.find('final number of phot. per tracked macropart.2') != -1:
                num_photon2 = float(row.split(" : ")[1].strip(' '))
            if row.find('de1=') != -1:
                energy_loss1 = float(row.split("=")[1].split(";")[0].strip(' '))*1e9
            if row.find('de2=') != -1:
                energy_loss2 = float(row.split("=")[1].split(";")[0].strip(' '))*1e9
    
    return lumi_ee_full, lumi_ee_peak, lumi_ee_geom, upsilon_max, num_pairs, num_photon1, num_photon2, energy_loss1, energy_loss2
