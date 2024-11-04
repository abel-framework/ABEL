import os
import abel.utilities.colors as cmaps

class CONFIG:

    # select cluster
    cluster_name = 'lumi' # lumi/betzy/LOCAL/etc.
    
    if cluster_name == 'lumi':
        #project_name = 'your_project_number_here'
        #project_name = 'project_465001375' # PLASMACOLLIDER project (E. Adli)
        project_name = 'project_465001379' # SPARTA project (C. A. Lindstrøm)
        partition_name_standard = 'standard-g'
        partition_name_small = 'small-g'
    elif cluster_name == 'betzy':
        project_name = 'your_project_number_here'
        #project_name = 'nn11003k'
        partition_name_standard = ''
        partition_name_small = ''
    elif cluster_name == 'LOCAL':
        project_name = ''
        partition_name_standard = ''
        partition_name_small = ''
    else:
        raise ValueError('cluster_name in CONFIG must be one of the valid options')
    
    ## ABEL STANDARD VALUES
    
    # plot width default
    plot_fullwidth_default = 18 # [cm]
    plot_width_default = 8 # [cm]

    # default colormap
    default_cmap = cmaps.FLASHForward # 'GnBu'
    
    
    ## ABEL DIRECTORIES
    
    # path to tracking data directory
    run_data_path = 'run_data/'
    
    # temporary directory
    temp_path = run_data_path + 'temp/'
    
    
    ## EXTERNAL CODE DIRECTORIES
    
    # common software path
    software_path = '/project/project_465001375/software/'
    
    # path to ELEGANT directory
    elegant_use_container = True
    elegant_path = os.path.join(software_path, 'elegant/')
    if elegant_use_container:
        bind_path = '/pfs/lustrep2/scratch/project_465001375'
        elegant_exec = 'singularity exec --bind ' + bind_path + ':' + bind_path + ' ' + elegant_path + 'elegant.sif '
        elegant_rpnflag = ''
    else:
        elegant_exec = elegant_path
        elegant_rpnflag = ' -rpnDefns=' + CONFIG.elegant_path + 'defns.rpn'
    
    # path to HiPACE++ directory
    hipace_path = os.path.join(software_path, 'hipace')
    hipace_binary = os.path.join(hipace_path,'build/bin/hipace')
    
    # path to GUINEA-PIG directory
    guineapig_path = os.path.join(software_path,'guinea-pig/bin/')
    
