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
    lumi_ee_full, lumi_ee_peak, lumi_ee_geom = guineapig_read_output(outputfile)
    
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
    
    return event
    

def guineapig_write_beam(beam, filename):
    
     # write beam phasespace to CSV
    M = np.matrix([beam.Es()/1e9, beam.xps(), beam.yps(), beam.xs()*1e9, beam.ys()*1e9, (beam.z_offset()-beam.zs())*1e6])
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(len(beam))):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i]])
    

def guineapig_read_output(outputfile):
    
    # string to search in file
    with open(outputfile, 'r') as fp:
        lines = fp.readlines()
        for row in lines:
            header = ' ------------- general results ------------ '
            if row.find(header) != -1:
                ind_header = lines.index(row)
                
                # calculate
                lumi_ee_geom = float((lines[ind_header+2]).split(" ")[2]) # [m^-2 per crossing]
                lumi_ee_full = float((lines[ind_header+3]).split(" ")[2]) # [m^-2 per crossing]
                lumi_ee_peak = float((lines[ind_header+4]).split(" ")[2]) # [m^-2 per crossing]
                
                return lumi_ee_full, lumi_ee_peak, lumi_ee_geom