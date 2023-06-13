import os

class CONFIG:
    
    
    ## OPAL STANDARD VALUES
    
    # plot width default
    plot_fullwidth_default = 18 # [cm]
    plot_width_default = 8 # [cm]
    
    
    ## OPAL DIRECTORIES
    
    # path to OPAL directory
    opal_path = str.replace(os.path.abspath(__file__), 'opal/' + os.path.basename(__file__), '')
    
    # path to tracking data directory
    run_data_path = 'run_data/'
    
    # temporary directory
    temp_path = opal_path + '.temp/'
    
    
    ## EXTERNAL CODE DIRECTORIES
    
    # path to ELEGANT directory
    elegant_path = opal_path + '../elegant/bin/'
    
    # path to HiPACE++ directory
    hipace_path = opal_path + '../hipace/'
    
    # path to GUINEA-PIG directory
    guineapig_path = opal_path + '../guinea-pig/bin/'
    