# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import os

class CONFIG:

    #List of paths to look for ABEL's configuration file, in order of preference.
    # If nothing was found, copy the one in templatepath into the first name in searchpath, and retry.
    _config_searchpath   = [os.path.join(os.path.expanduser('~'), ".abelconfig.toml"),]
    _config_templatepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"abelconfig.toml")

    @classmethod
    def initialize(cls, verbose=False):
        "Used to initialize the CONFIG class from abel/__init__.py after import, since we later access it as CONFIG.varname"
        if len(CONFIG._config_searchpath) == 0:
            raise ValueError("_config_searchpaths is empty; something is wrong")
        
        configFile = None
        for p in cls._config_searchpath:
            if os.path.isfile(p):
                if verbose:
                    print("Loading ABEL config from '"+p+"'")
                configFile = p
        if configFile == None:
            print("Copying a template config into '",cls._config_searchpath[0],"'")
            import shutil
            shutil.copy2(cls._config_templatepath, cls._config_searchpath[0])
            print("Now you should check and maybe edit this file, and then import ABEL again.")
            raise ValueError(f"ABEL configuration not found; creating default in {cls._config_searchpath[0]}.")

        #Load and parse the data from the toml file
        import tomllib
        with open(configFile, 'rb') as cf:
            cfdata = tomllib.load(cf)
        if verbose:
            print(cfdata)

        #TODO: Keep track of which variables from the toml have been used,
        #      and which haven't, then print a warning at the end

        def parsePath(pathData):
            "Helper function to parse abel's toml paths inside initialize()"
            if type(pathData) == str:
                return pathData
            elif type(pathData) == list:
                #Search for keywords and replace
                for i in range(len(pathData)):
                    p = pathData[i]
                    if type(p) != str:
                        print()
                        raise TypeError(f'Expected each element of pathData to be a string, got {type(pathData)} for {pathData}')
                    if len(p) > 1:
                        if p[0] == '$':
                            if p[1] == '$':
                                k = p[2:]
                                if not k in os.environ:
                                    raise KeyError(f"Invalid environment variable '{k}'")
                                p = os.environ[k]
                            elif p[1:] == 'run_data_path':
                                p = cls.run_data_path
                            elif p[1:] == 'software_path':
                                p = cls.software_path
                            elif p[1:] == 'hipace_path':
                                p = cls.hipace_path
                            else:
                                raise ValueError("Unknown path keyword '"+p+"'")
                            pathData[i] = p
                return os.path.join(*pathData) #Use the 'splat' operator *

        # Parse

        # Compute cluster configuration
        cls.cluster_name = cfdata['cluster']['cluster_name']
        #Load default cluster settings
        if cls.cluster_name == 'lumi':
            #cls.project_name = 'project_465001375' # PLASMACOLLIDER project (E. Adli)
            cls.project_name = 'project_465001379' # SPARTA project (C. A. Lindstrøm)
            cls.partition_name_standard = 'standard-g'
            cls.partition_name_small = 'small-g'
            cls.partition_name_devel = 'dev-g'
        elif cls.cluster_name == 'betzy':
            cls.project_name = 'your_project_number_here'
            #cls.project_name = 'nn11003k'
            cls.partition_name_standard = ''
            cls.partition_name_small = ''
            cls.partition_name_devel = ''
        elif cls.cluster_name == 'LOCAL':
            cls.project_name = ''
            cls.partition_name_standard = ''
            cls.partition_name_small = ''
            cls.partition_name_devel = ''
        else:
            raise ValueError("cluster_name in CONFIG must be one of the valid options, got '"+cls.cluster_name+"'")
        #Optional overrides
        if 'project_name' in cfdata['cluster']:
            cls.project_name = cfdata['cluster']['project_name']
        if 'partition_name_standard' in cfdata['cluster']:
            cls.partition_name_standard = cfdata['cluster']['partition_name_standard']
        if 'partition_name_small' in cfdata['cluster']:
            cls.partition_name_small = cfdata['cluster']['partition_name_small']
        if 'partition_name_devel' in cfdata['cluster']:
            cls.partition_name_devel = cfdata['cluster']['partition_name_devel']

        # ABEL plot defaults
        import abel.utilities.colors as cmaps
        cls.plot_fullwidth_default = cfdata['defaults']['plot_fullwidth_default']
        cls.plot_width_default     = cfdata['defaults']['plot_width_default']
        cls.default_cmap           = cmaps.get_cmap_by_name(cfdata['defaults']['default_cmap'])

        # Directories
        cls.run_data_path  = parsePath(cfdata['directories']['run_data_path'])
        cls.temp_path      = parsePath(cfdata['directories']['temp_path'])

        # External codes
        cls.software_path  = parsePath(cfdata['external_codes']['software_path'])

        # External codes - ELEGANT
        cls.elegant_use_container = cfdata['external_codes']['elegant']['elegant_use_container']
        cls.elegant_path   = parsePath(cfdata['external_codes']['elegant']['elegant_path'])
        if cls.elegant_use_container:
            cls.bind_path  = parsePath(cfdata['external_codes']['elegant']['bind_path'])
            cls.elegant_exec = 'singularity exec --bind ' + cls.bind_path + ':' + cls.bind_path + ' ' + cls.elegant_path + 'elegant.sif '
            cls.elegant_rpnflag = ''
        else:
            cls.elegant_exec = cls.elegant_path
            cls.elegant_rpnflag = ' -rpnDefns=' + CONFIG.elegant_path + 'defns.rpn'
        #Optional overrides
        if 'elegant_exec' in cfdata['external_codes']['elegant']:
            cls.elegant_exec = cfdata['external_codes']['elegant']['elegant_exec']
        if 'elegant_rpnflag' in cfdata['external_codes']['elegant']:
            cls.elegant_rpnflag = cfdata['external_codes']['elegant']['elegant_rpnflag']

        # External codes - HIPACE++
        cls.hipace_path    = parsePath(cfdata['external_codes']['hipace']['hipace_path'])
        cls.hipace_binary  = parsePath(cfdata['external_codes']['hipace']['hipace_binary'])

        # External codes - GuineaPig
        cls.guineapig_path = parsePath(cfdata['external_codes']['guineapig']['guineapig_path'])

    @classmethod
    def printCONFIG(cls):
        ls = dir(cls)
        for d in ls:
            if d.startswith('_'):
                continue
            elif d == 'printCONFIG' or d=='initialize':
                continue

            att = getattr(cls,d)
            if type(att) == str:
                print(f'CONFIG.{d} = "{att}"')
            else:
                print(f'CONFIG.{d} = {att}')
