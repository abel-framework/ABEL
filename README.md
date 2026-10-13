# ABEL: the Adaptable Beginning-to-End Linac framework

<p>
<img width="100" height="100" alt="abel_logo" src="https://github.com/user-attachments/assets/23247a60-ab84-42ef-a647-c899f6932766" style="float:left; padding-left:10px; padding-right:10px"/>

The ABEL simulation framework is a particle-tracking framework for multi-element particle accelerators (such as plasma-accelerator linacs, colliders, experimental test facilities, etc.), implemented at varying levels of complexity, for fast investigations or optimizations. As a systems code, it can be used for physics simulations as well as generating (and optimizing for) cost estimates.
<br><br>
</p>

# Installation of ABEL
The project name is `abel-framework`, which contains the `abel` package with various sub-packages.
The project homepage is: https://github.com/abel-framework/ABEL/

ABEL needs Python v3.11, due to requirements from some dependencies.

## 1. Creating a Pyton environment for ABEL:

### Using `venv`
Make and activate a new Python (version 3.11) virtual environment for ABEL:\
`python3.11 -m venv your_abel_venv` and `source your_abel_venv/bin/activate`

### Using `conda`
If not already done, setup `conda-forge`:\
`conda config --add channels conda-forge` and `conda config --set channel_priority strict`

Make and activate a new Python conda environment for ABEL:\
`conda create -p abel-2026 python=3.12` and `conda activate /home/kyrsjo/CONDA/abel-2026`

You may consider to then manually install some packages:\
`conda install numpy scipy numba matplotlib tqdm PyQt6 pyqtgraph toml dill`

## 2. Installing `abel` into the Python environment

### Standard installation from PyPi
After preparing the Python environment, simply run:\
`pip install abel-framework`\
This will also install any missing dependencies in the right versions, including Wake-T, RF-Track, CLICopti, ax-platform, and impactx.

To remove ABEL from the python environment, run `pip uninstall abel-framework`.

### Editable installation from GitHub with `pip` (for development)
To install a specific version from GitHub, e.g. for development, you first need to clone the repository to a local folder:\
`git clone https://github.com/abel-framework/ABEL`\
Note that if you intend to contribute (push) to ABEL, cloning with SSH instead of HTTPS is reccomended.

Once the repository has been cloned, in your target python environment, run\
`pip install -e path-to-ABEL` where `path-to-ABEL` is the folder you have cloned ABEL to.

If you want to be able to modify ABEL without uninstalling and reinstalling, you can instead run\
 `pip install -e path-to-abel`\
 Changes to the files in the folder `src\abel` will be reflected in the installed package, as if you had put it into your `$PYTHONPATH`.

To remove ABEL, run `pip uninstall abel-framework`.

## Configuration of ABEL
To use ABEL, you must configure it. This is done with the file `.abelconfig.toml`, which is automatically created in your home directory the first time you import ABEL.
Edit this file with your text editor to tell ABEL where to find tools such as [ELEGANT](https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html), [HIPACE++](https://github.com/Hi-PACE/hipace/), and [GUINEA-PIG](https://gitlab.cern.ch/clic-software/guinea-pig), as well as configure it for your computing cluster, if needed.

Comments in the file explain how to edit it. The configuration file uses the [TOML format](https://en.wikipedia.org/wiki/TOML), which is a simple text file similar to `.ini`, but more well defined.

Please do not edit the template file `abelconfig.toml` or `CONFIG.py` in the source code folder.

The loaded configuration of ABEL is printed to the terminal when ABEL starts, along with the name of the config file it has loaded.

## Unit tests
Unit tests are implemented with `pytest`; to run the tests on an installed version of ABEL please run
```
pytest -v
```
from the root folder of ABEL.
It is also possible to run single tests by name, for example: `pytest -v tests/test_init.py::testCore_init`

The tests are stored in in the `tests` subdirectory, in files with names starting with `test_`.
In these files, the functions with names starting with `test_` are represent one test; if it makes it to the end without any of the asserts triggering and all the expected exceptions happening, the test has PASSED.
The test functions are also annotated with `@pytest.mark.MARKNAME`.

Pytest is configured in the `[tool.pytest.ini_options]` section of `pyproject.toml`. This especially defines the "markers", which are named groups of tests that can be specified to run using `pytest -v -m MARKNAME`.

When the tests succeed, no output (except `testfile::testfile PASSED`) is printed. If a test fails, a traceback and the printouts of that test are printed. If many tests fail, this can be very verbose.

## References
Please cite the following when referring to ABEL or using ABEL simulations for publications:

[1] J. B. B. Chen et al., _ABEL: The Adaptable Beginning-to-End Linac simulation framework_, [Proceedings of IPAC 2025 (Taipei, Taiwan, 2025), pp. 1438-1441](https://meow.elettra.eu/81/pdf/TUPS012.pdf).

## Acknowledgements
This work was supported by the European Research Council (project [SPARTA](https://www.mn.uio.no/fysikk/english/research/projects/staging-of-plasma-accelerators-for-timely-applications/), Grant No. [101116161](https://doi.org/10.3030/101116161)) and the Research Council of Norway (Grant No. [313770](https://prosjektbanken.forskningsradet.no/project/FORISS/313770) and [353317](https://prosjektbanken.forskningsradet.no/project/FORISS/353317)).


## Mailing list
[Subscribe](https://sympa.uio.no/fys.uio.no/info/abel-framework) to the mailing list abel-framework@fys.uio.no for updates.