import os
import abel.utilities.colors as cmaps

class CONFIG:
    
    ## ABEL STANDARD VALUES
    
    # plot width default
    plot_fullwidth_default = 18 # [cm]
    plot_width_default = 8 # [cm]

    # default colormap
    default_cmap = cmaps.FLASHForward # 'GnBu'
    
    
    ## ABEL DIRECTORIES
    
    # path to ABEL directory
    abel_path = str.replace(os.path.abspath(__file__), 'abel/' + os.path.basename(__file__), '')
    
    # path to tracking data directory
    run_data_path = 'run_data/'
    
    # temporary directory
    temp_path = run_data_path + 'temp/'
    
    
    ## EXTERNAL CODE DIRECTORIES
    
    # common software path
    software_path = '/project/project_465000445/software/'
    
    # path to ELEGANT directory
    elegant_use_container = True
    elegant_path = software_path + 'elegant/'
    if elegant_use_container:
        bind_path = '/pfs/lustrep2/scratch/project_465000445'
        elegant_exec = 'singularity exec --bind ' + bind_path + ':' + bind_path + ' ' + elegant_path + 'elegant.sif '
        elegant_rpnflag = ''
    else:
        elegant_exec = elegant_path
        elegant_rpnflag = ' -rpnDefns=' + CONFIG.elegant_path + 'defns.rpn'
    
    # path to HiPACE++ directory
    hipace_path = software_path + 'hipace/'
    
    # path to GUINEA-PIG directory
    guineapig_path = software_path + 'guinea-pig/bin/'
    
