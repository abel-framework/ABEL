# ABEL: the Adaptable Beginning-to-End Linac simulation framework

The ABEL simulation framework is a particle-tracking framework for plasma-accelerator linacs, implemented at varying levels of complexity, for fast optimization.

## Installation with pip
1. Clone the repository to a local folder
2. In your target python environment, run `pip install path-to-ABEL` where `path-to-ABEL` is where you have cloned ABEL to.
   If you want to be able to modify ABEL without uninstalling and reinstalling, you can run `pip install -e path-to-abel`, and the `abel`
   folder in your local clone will effectively be put into your `$PYTHONPATH`.

To remove ABEL, run `pip uninstall abel`.

## Configuration of ABEL
To use ABEL, you must configure it. This is done with the file `.abelconfig.toml`, which is automatically created in your home directory the first time you import ABEL.
Edit this file with your text editor to tell ABEL where to find tools such as ELEGANT, HIPACE, and GUINEAPIG, as well as configure it for your computing cluster, if needed.

How to edit it is explained with comments in the file. It uses the file format "TOML", which is a simple text file similar to .ini, but better defined.

Please do not edit the template file `abelconfig.toml` or `CONFIG.py` in the source code folder.

The as-loaded configuration of ABEL is printed to the terminal when abel starts, along with the name of the config file it has loaded.
