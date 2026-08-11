# Pytorch Surrogate Model (from ImpactX)

The scripts here are part of an adaptation of the ImpactX pytorch surrogate model example (https://impactx.readthedocs.io/en/latest/usage/examples/pytorch\_surrogate\_model/README.html) into the ABEL 
framework. All scripts were written by Codex, and the names should match the original ImpactX script names with the addition of the string `_abel` at the end. The details of this example are explained in the 
above link, and so for brevity I will point the user to that resource rather than explain it myself. 

## Run Instructions

The main script is `run_ml_surrogate_15_stage_abel.py`, which sets up the initial beam, loads the surrogate model for the plasma stages, defines the lattice geometry, and runs the particle tracking. To produce 
an output (hdf5) file, utilize the argument `--save-h5` when running the script: 

```
python run_ml_surrogate_15_stage_abel.py --save-h5 /path/to/output/file.h5
```

The output of the run is analyzed using the `analyze_ml_surrogate_15_stage_abel.py` script, which prints the initial and final beam moments:

```
python analyze_ml_surrogate_15_stage_abel.py --file_path /path/to/output/file.h5
```

Visualization is done via the `visualize_ml_surrogate_15_stage_abel.py` script, which produces three plots: electron beam moments at the 15 different LPA surrogate stages, the initial phase-space distributions, and the final phase-space distributions. 

```
python visualize_ml_surrogate_15_stage_abel.py --file-path /path/to/output/file.h5 --save-png
```

The `--save-png` argument will produce the three output plots in .png format. 
