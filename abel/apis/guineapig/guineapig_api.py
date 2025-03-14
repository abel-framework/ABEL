import time, os, uuid, shlex, subprocess, csv
import numpy as np
from abel import CONFIG, Event

def guineapig_run(inputfile, beam1, beam2):
    
    # make temporary output file
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
    os.mkdir(tmpfolder)
    outputfile = tmpfolder + "/output.ref"
    
    # make temporary beam files
    beamfile1 = tmpfolder + "/inputbeam1.ini"
    beamfile2 = tmpfolder + "/inputbeam2.ini"
    guineapig_write_beam(beam1, beamfile1)
    guineapig_write_beam(beam2, beamfile2)

    # run GUINEA-PIG
    cmd = CONFIG.guineapig_path + 'guinea default default ' + outputfile + ' --el_file=' + beamfile1 + ' --pos_file=' + beamfile2 + ' --acc_file=' + inputfile
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    
    # parse outputs
    lumi_ee_full, lumi_ee_peak, lumi_ee_geom, upsilon_max, num_pairs, num_photon1, num_photon2, energy_loss1, energy_loss2 = guineapig_read_output(outputfile)
    
    # remove temporary files and folders
    os.remove(outputfile)
    os.remove(beamfile1)
    os.remove(beamfile2)
    os.rmdir(tmpfolder)
    
    # make event object
    event = Event(beam1, beam2)
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
    

def guineapig_write_beam(beam, filename):
    
     # write beam phasespace to CSV
    M = np.matrix([beam.Es()/1e9, beam.xps(), beam.yps(), beam.xs()*1e9, beam.ys()*1e9, (beam.z_offset()-beam.zs())*1e6])
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(len(beam))):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i]])
    

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
