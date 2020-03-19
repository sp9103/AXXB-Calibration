# AXXB-Calibration
For Robot-Scanner calibration script

Package
*  Numpy
*  Scipy
*  Pyinstaller (for create exe file)

How to create execution file
* pyinstaller AXXBSolver.py --hidden-import scipy.linalg.logm --path "[PYTHON_DIR]\\Lib\\site-packages\\scipy\\extra-dll" -F
