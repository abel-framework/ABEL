# ABEL: the Advanced Beginning-to-End Linac simulation framework

The ABEL simulation framework is a particle-tracking framework for plasma-accelerator linacs, implemented at varying levels complexity, for fast optimization.

## Installation with pip
1. Clone the repository to a local folder
2. Edit the file abel/config.py. In particular, you'll need to install HiPACE and tell it where it is located; ABEL needs the "tools" python module in their source folder.
3. In your target python environment, run `pip install path-to-ABEL` where `path-to-ABEL` is where you have cloned ABEL to.
   If you want to be able to modify ABEL without uninstalling and reinstalling, you can run `pip install -e path-to-abel`, and the `abel`
   folder in your local clone will effectively be put into your `$PYTHONPATH`.

To remove abel, run `pip uninstall abel`.