import os

class CONFIG:
    
    
    ## ABEL STANDARD VALUES
    
    # plot width default
    plot_fullwidth_default = 18 # [cm]
    plot_width_default = 8 # [cm]
    
    
    ## ABEL DIRECTORIES
    
    # path to ABEL directory
    abel_path = str.replace(os.path.abspath(__file__), 'abel/' + os.path.basename(__file__), '')
    
    # path to tracking data directory
    run_data_path = 'run_data/'
    
    # temporary directory
    temp_path = run_data_path + '.temp/'
    
    
    ## EXTERNAL CODE DIRECTORIES
    
    # common software path
    software_path = '/mn/fys-server1/a8/carlal/software/'
    
    # path to ELEGANT directory
    elegant_path = software_path + 'elegant/usr/bin/'
    
    # path to HiPACE++ directory
    hipace_path = software_path + 'hipace/'
    
    # path to GUINEA-PIG directory
    guineapig_path = software_path + 'guinea-pig/bin/'
    