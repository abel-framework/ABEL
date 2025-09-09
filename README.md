# ABEL: the Adaptable Beginning-to-End Linac framework

The ABEL simulation framework is a particle-tracking framework for multi-element particle accelerators (such as plasma-accelerator linacs, colliders, experimental test facilities, etc.), implemented at varying levels of complexity, for fast investigations or optimizations. As a systems code, it can be used for physics simulations as well as generating (and optimizing for) cost estimates.

## Installation with `pip`
1. (Optional) Make and activate a new Python (version 3.11) virtual environment for ABEL: `python3.11 -m venv your_abel_venv`
2. Clone the repository to a local folder, e.g., `git clone https://github.com/abel-framework/ABEL`
3. In your target python environment, run `pip install path-to-ABEL` where `path-to-ABEL` is where you have cloned ABEL to.
   If you want to be able to modify ABEL without uninstalling and reinstalling, you can run `pip install -e path-to-abel`, and the `abel`
   folder in your local clone will effectively be put into your `$PYTHONPATH`.

To remove ABEL, run `pip uninstall abel`.

## Configuration of ABEL
To use ABEL, you must configure it. This is done with the file `.abelconfig.toml`, which is automatically created in your home directory the first time you import ABEL.
Edit this file with your text editor to tell ABEL where to find tools such as ELEGANT, [HIPACE++](https://github.com/Hi-PACE/hipace/), and GUINEAPIG, as well as configure it for your computing cluster, if needed.

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

## Acknowledgements
This work was supported by the European Research Council (project [SPARTA](https://www.mn.uio.no/fysikk/english/research/projects/staging-of-plasma-accelerators-for-timely-applications/), Grant No. [101116161](https://doi.org/10.3030/101116161)) and the Research Council of Norway (Grant No. [313770](https://prosjektbanken.forskningsradet.no/project/FORISS/313770) and [353317](https://prosjektbanken.forskningsradet.no/project/FORISS/353317)).
