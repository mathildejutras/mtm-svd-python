# MTM-SVD method in python

This repository contains scripts to apply the **MTM-SVD analysis method** to climate data and model outputs. It is a direct adaptation of the Matlab script developed by M. Correa-Ramirez. 

![Example](/images/example.jpg)

This script was adapted by Mathilde Jutras at McGill University, Canada

**Copyright (C) 2020, Mathilde Jutras**

The script is available under the *GNU General Public License v3.0*.
It may be used, copied, or redistributed as long as it is cited as follows:

A description of the theoretical basis of the MTM-SVD toolbox and some
implementation details can be found in:
*Correa-Ramirez, M. & S. Hormazabal, 2012. "MultiTaper Method-Singular Value
Decomposition (MTM-SVD): Variabilidad espacio–frecuencia de las
fluctuaciones del nivel del mar en el Pacífico Suroriental",
Lat. Am. J. Aquat. Res.*

The functions of the Matlab toolbox are based on the MTM-SVD FORTRAN functions
developed by Michael Mann and Jeffrey Park
(http://holocene.meteo.psu.edu/Mann/tools/MTM-SVD/),
and the Matlab toolbox can be found on the same website.

The main script is contained in mtm-svd-python.py, and the required functions can be found in mtm_functions.py.
