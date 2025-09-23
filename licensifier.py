#!/usr/bin/env python3

# Copyright 2025 The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

#This is the licensifier - it crawls the ABEL source tree,
# and inserts the license blurb if it is missing.

AUTHORS = "C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli"
LICENSE = "GPL-3.0-or-later"

BLURB =\
f"""# Copyright 2022-, The ABEL Authors
# Authors: {AUTHORS}
# License: {LICENSE}"""

BLURB_RE = [r"(?i)#\.*copyright", r"(?i)GNU General Public License"]

SEARCH_FOLDERS = ["abel", "tests"]
SEARCH_FILETYPES_RE = [r".*\.py",]

SEARCH_EXPLICIT_FILES = ['licensifier.py',]
EXCLUDE_EXPLICIT_FILES = []

import os
import re
import sys

has_inexact_blurb_files = []

def treat_file(filename, doIt=False):
    print("checking inside ... ", end='')

    fi_ = open(filename,'r')
    fi = fi_.read()
    fi_.close()

    # Look for a pre-existing BLURB within the file
    has_blurb = False
    for r in BLURB_RE:
        if re.search(r, fi):
            has_blurb = True
            print('has_blurb ...', end='')

    #Look for a pre-existing exactly matching BLURB        
    has_exact_blurb = False
    if BLURB in fi:
        has_exact_blurb = True
        print("has_exact_blurb ...", end='')
        return()

    if has_blurb and not has_exact_blurb:
        has_inexact_blurb_files.append(filename)
        return()

    #TODO: Handle update of core authors and year
    
    #Blurbless file!
    #INSERT!

    #After hashbang
    BLURB_ = BLURB
    if fi.startswith('#!'):
        fi_start = fi.index('\n')+1
        print("\t skip-hashbang",end='')
        BLURB_ = "\n" + BLURB + 2*"\n"
    else:
        fi_start = 0
        BLURB_ = BLURB + 2*"\n"
    
    fi = fi[:fi_start] + BLURB_ + fi[fi_start:]

    print("EDIT:")
    print('"""')
    print (fi[0:len(BLURB_)*2])
    print('...')
    print('"""')

    if doIt:
        fi_ = open(filename,'w')
        fi_.write(fi)
        fi_.close()
        print("\t WRITTEN!")
    else:
        print("\t (dry-run mode)")
    

if __name__ == "__main__":

    print("Usage:")
    print(sys.argv[0] + " [--doit] [filename1 filename2 ...]")
    print("Run with '--doit' in order to write modifications to files, by default we run in dry-run mode.")
    print("One or more file paths can be specified, if so it will only modify these.")
    print("By default, all files passing the regexes in SEARCH_FILETYPES_RE in SEARCH FOLDERS, but not in EXCLUDE_EXPLICIT_FILES will be analysed/treated")
    print("Also by default, all files in SEARCH_EXPLICIT_FILES will also be analysed/treated")
    print()
    print("Tip: Before running with --doit, commit all your changes.")
    print("This way you can undo the licensifier changes by running `git checkout .` in the repository root.")
    print()
    
    print("BLURB to be added:")
    print('"""')
    print(BLURB)
    print('"""')
    print()

    doIt = False
    cmdline_files = []
    if len(sys.argv) >= 2:
        i = 1
        if sys.argv[i] == "--doit":
            doIt = True
            i += 1
        while i < len(sys.argv):
            cmdline_files.append(sys.argv[i])
            i += 1
    if len(cmdline_files) > 0:
        print("Searching files explicitly given at commandline:")
        for fp in cmdline_files:
            print(f"'{fp}' : ", end='')
            if not os.path.exists(fp):
                print("MISSING!")
                continue

            print("\n\t",end='')
            treat_file(fp, doIt)
            print()
        print("Skipping SEARCH_FOLDERS and SEARCH_EXPLICIT_FILES since files were given on command line.")
        exit(0)
    
    print('Searching SEARCH_FOLDERS:')
    for root_folder in SEARCH_FOLDERS:
        #Iterate over search root folders from SEARCH_FOLDERS
        for root, folders, files in os.walk(root_folder):
            #Iterate over folders from root_folder,
            # and for each sub/folder make a list of folders and files.
            
            #print(root, dirs, files)

            for f in files:
                fp = os.path.join(root,f)
                print( f"'{fp}' : ", end='')

                #Check if the filetype is listed in SEARCH_FILETYPES_RE
                toCheck = False
                for r in SEARCH_FILETYPES_RE:
                    if re.fullmatch(r, f):
                        toCheck = True
                        print("filename match...", end='')
                    else:
                        print("no filename match.", end='')
                if toCheck and not (fp in EXCLUDE_EXPLICIT_FILES):
                    print("\n\t",end='')
                    treat_file(fp, doIt)
                elif fp in EXCLUDE_EXPLICIT_FILES:
                    print("Explicitly excluded!", end='')
                print()

    print()
    print("Searching SEARCH_EXPLICIT_FILES:")
    for fp in SEARCH_EXPLICIT_FILES:
        print(f"'{fp}' : ", end='')
        if not os.path.exists(fp):
            print("MISSING!")
            continue

        print("\n\t",end='')
        treat_file(fp, doIt)
        print()
            

                    
    
    
                
