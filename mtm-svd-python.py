# Script for MultiTaper Method-Singular Value Decomposition (MTM-SVD) in python
#
# ------------------------------------------------------------------
#
# This script is a direct adaptation of the Matlab toolbox developed by
# Marco Correa-Ramirez and Samuel Hormazabal at 
# Pontificia Universidad Catolica de Valparaiso
# Escuela de Ciencias del Mar, Valparaiso, Chile
# and is available through 
# http://www.meteo.psu.edu/holocene/public_html/Mann/tools/tools.php
#
# This script was adapted by Mathilde Jutras at McGill University, Canada
# Copyright (C) 2020, Mathilde Jutras
# and is available under the GNU General Public License v3.0
# 
# The script may be used, copied, or redistributed as long as it is cited as follow:
# Mathilde Jutras. (2020, July 6). mathildejutras/mtm-svd-python: v1.0.0-alpha (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3932319
#
# This software may be used, copied, or redistributed as long as it is not 
# sold and that this copyright notice is reproduced on each copy made. 
# This routine is provided as is without any express or implied warranties.
#
# Questions or comments to:
# M. Jutras, mathilde.jutras@mail.mcgill.ca
#
# Last update:
# July 2020
#
# ------------------------------------------------------------------
#
# The script is structured as follows:
#
# In the main script is found in mtm-svd-python.py
# In the first section, the user can load the data,
# assuming the outputs are stored in a netcdf format.
# In the secton section, functions are called to calculate the spectrum
# The user will then be asked for which frequencies he wants to plot 
# the spatial patterns associated with the variability.
# In the third section, the spatial patterns are plotted and saved
#
# The required functions are found in mtm_functions.py
#
# ------------------------------------------------------------------
#

from mtm_functions import *
import xarray as xr
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------
# 1) Load the data
# -----------------

path = ...

files = listdir(path)
files.sort()

# Select the depth, in m
d = 300 

print('Load data...')

dsl = xr.open_dataset(files[0])
dt = 1./len(dsl.time) # time step in years
lon = dsl.NAME_LON_VARIABLE
lat = dsl.NAME_LAT_VARIABLE

ds = xr.open_mfdataset(files)
var = ds.NAME_VARIABLE 
var = var.sel(NAME_DEPTH_VARIABLE=d, method='nearest').values

# Plot map of the variable
xgrid, ygrid = np.meshgrid(lon,lat)
plt.pcolor(xgrid, ygrid, var[0,:,:], cmap='jet')
cbar=plt.colorbar()
plt.show()

# -------------------
# 2) Compute the LVF
# -------------------

print('Apply the MTM-SVD...')

# Slepian tapers
nw = 2; # bandwidth
kk = 3; # number of orthogonal windows

# Reshape the 2d array to a 1d array
o2ts = o2.reshape((o2.shape[0],o2.shape[1]*o2.shape[2]), order='F')
p, n = o2ts.shape

# Compute the LFV
[freq, lfv] = mtm_svd_lfv(o2ts,nw,kk,dt)

# Compute the confidence intervals
niter = 1000 # minimum of 1000 iterations
sl = [.9]
[conffreq, conflevel] = mtm_svd_conf(o2ts,nw,kk,dt,niter,sl)

# Plot the spectrum

plt.semilogx(freq,lfv)
plt.semilogx(conffreq,conflevel[0,:],'--r')
xt = [1./10., 1./5., 1./2.]
plt.xlim([1./60., 2.])
plt.xticks(xt, [1./each for each in xt])
plt.xlabel('Period [year]') ; plt.ylabel('Variance')
plt.title('LVF at %i m'%d)
plt.savefig('Figs/spectrum_%s_%im.jpg'%(model,d))
plt.clf()

# Display the plot to allow the user to choose the frequencies associated with peaks
plt.plot(freq, lfv)
plt.plot(conffreq, conflevel[0,:], '--', c='grey')
plt.show()
plt.clf()

fo = [float(each) for each in input('Enter the frequencies for which there is a significant peak and for which you want to plot the map of variance (separated by commas, no space):').split(',')]

# --------------------------------
# 3) Reconstruct spatial patterns
# --------------------------------

# Select frequency(ies) (instead of user-interaction selection)
#fo = [0.02, 0.05, 0.15, 0.19, 0.24, 0.276, 0.38] 

# Calculate the reconstruction

vexp, totvarexp, iis = mtm_svd_recon(o2ts,nw,kk,dt,fo)

# Plot the map for each frequency peak

for i in range(len(fo)):

	RV = np.reshape(vexp[i],xgrid.shape, order='F')

	fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[1,3]},figsize=(5,7))

	ax1.plot(freq, lfv)
	ax1.plot(conffreq, conflevel[0,:], '--', c='grey')
	ax1.plot(freq[iis[i]],lfv[iis[i]],'r*',markersize=10)
	ax1.set_xlabel('Frequency [1/years]')
	ax1.set_title('LVF at %i m'%d)

	pc = ax2.pcolor(xgrid, ygrid, RV, cmap='jet', vmin=0, vmax=50) 
	cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
	cbar.set_label('Variance')
	ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

	plt.tight_layout()
	plt.savefig('Figs/peak_analysis_%s_%im_%.2fyrs.jpg'%(model,d,1./fo[i]))
	#plt.show()
	plt.clf()
