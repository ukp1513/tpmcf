from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from astropy.stats import bootstrap
import datetime
import os
from random import sample 
import pandas as pd
from sklearn.cluster import KMeans
import healpy as hp
import copy
from shapely.geometry import Polygon, Point

def _ra_dec_to_cartesian(ra, dec):

	ra_rad = np.radians(ra)
	dec_rad = np.radians(dec)
	x = np.cos(ra_rad) * np.cos(dec_rad)
	y = np.sin(ra_rad) * np.cos(dec_rad)
	z = np.sin(dec_rad)
	return x, y, z
    

def _assign_weights(ra, dec, mask_df, nside):

	# Convert RA, Dec to HEALPix pixel indices
	theta = np.radians(90.0 - dec)  # convert to polar angle in radians
	phi = np.radians(ra)            # convert to azimuthal angle in radians
	pixels = hp.ang2pix(nside, theta, phi)

	# Create a dictionary of pixel weights from the mask
	pixel_weights = dict(zip(mask_df['PIXEL'], mask_df['SIGNAL']))

	# Assign weights to each data point
	weights = np.array([pixel_weights.get(pixel, 0) for pixel in pixels])

	return weights
	
	
def _weighted_kmeans(ra, dec, weights, n_patches):

	# Convert RA-Dec to Cartesian coordinates
	x, y, z = _ra_dec_to_cartesian(ra, dec)
	coords = np.vstack((x, y, z)).T

	# Repeat coordinates based on weights
	weighted_coords = np.repeat(coords, weights.astype(int), axis=0)

	# Perform k-means clustering
	kmeans = KMeans(n_clusters=n_patches, random_state=42)
	labels = kmeans.fit_predict(weighted_coords)

	# Assign original data points to clusters based on nearest cluster centers
	original_labels = kmeans.predict(coords)

	# Group RA-Dec by patch
	patches = []
	for i in range(n_patches):
		indices = np.where(original_labels == i)[0]
		ra_patch = ra.iloc[indices]
		dec_patch = dec.iloc[indices]
		weights_patch = weights.iloc[indices]
		patches.append({'RA': ra_patch, 'Dec': dec_patch, 'weights': weights_patch, 'indices': indices})

	return patches

def load_mask_from_fits(fits_file):
	with fits.open(fits_file) as hdul:
		data = hdul[1].data  # Assuming mask data is in the second HDU extension (index 1)

	pixel = data['PIXEL']
	signal = data['SIGNAL']

	mask_df = pd.DataFrame({'PIXEL': pixel, 'SIGNAL': signal})

	return mask_df

def makeJkSamples_equalArea_kmeansClustering(real_table, rand_table, njacks, realracol, realdeccol, randracol, randdeccol, mask_file, plotSamples=False, plotPatches=True):

	mask_df = load_mask_from_fits(mask_file)
	
	nside = 4096  # Adjust based on your mask resolution
	real_table['weights'] = _assign_weights(real_table['RA'], real_table['DEC'], mask_df, nside)
	
	n_patches = njacks
	patches = _weighted_kmeans(real_table[realracol], real_table[realdeccol], real_table['weights'], n_patches)
	
	if(plotPatches):
		for i, patch in enumerate(patches):
			plt.scatter(patch['RA'], patch['Dec'], s=1)
		plt.savefig("JKpatches.png", dpi=300, bbox_inches = 'tight')
		plt.close()

	jackknife_samples = []
	
	for i in range(n_patches):
		patches_tmp = copy.deepcopy(patches)
		patch_index_to_remove = i

		removed_patch = patches_tmp.pop(patch_index_to_remove)

		remaining_indices = np.concatenate([patch['indices'] for patch in patches_tmp])
		realdata_jknow = real_table.iloc[remaining_indices].reset_index(drop=True)
		randdata_jknow = rand_table.iloc[remaining_indices].reset_index(drop=True)
		
		if(plotSamples):
			plt.scatter(realdata_jknow[realracol], realdata_jknow[realdeccol], s=1)
			plt.scatter(randdata_jknow[randracol], randdata_jknow[randdeccol], s=1)
			plt.show()
			plt.close()
		
		jackknife_samples.append((realdata_jknow, randdata_jknow))

	return jackknife_samples


def makeJkSamples(real_table, rand_table, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False):

	n_d = len(real_table)

	n_r = len(rand_table)

	jack_nr = 0

	ra_min, ra_max = min(rand_table[randracol]), max(rand_table[randracol])
	dec_min, dec_max = min(rand_table[randdeccol]),max(rand_table[randdeccol])


	# CREATING JACKKNIFE SAMPLES
	jackknife_samples = []

	for jk_r in range(njacks_ra):
		for jk_c in range(njacks_dec):
			jack_nr +=1 

			raMinNow = ra_min + (jk_r * (ra_max-ra_min)/njacks_ra)
			raMaxNow = raMinNow + (ra_max-ra_min)/njacks_ra

			decMinNow = dec_min + (jk_c * (dec_max-dec_min)/njacks_dec)
			decMaxNow = decMinNow + (dec_max-dec_min)/njacks_dec

			real_jk= real_table[((real_table[realracol] <= raMinNow) | (real_table[realracol] >= raMaxNow)) | ((real_table[realdeccol] <= decMinNow) | (real_table[realdeccol] >= decMaxNow))]
			rand_jk= rand_table[((rand_table[randracol] <= raMinNow) | (rand_table[randracol] >= raMaxNow)) | ((rand_table[randdeccol] <= decMinNow) | (rand_table[randdeccol] >= decMaxNow))]
			if(plot):
				plt.scatter(real_jk[realracol], real_jk[realdeccol], s=1, color='blue')
				plt.show()

			jackknife_samples.append((real_jk, rand_jk))
	
	return jackknife_samples
	
def giveJkSample(sample_nr, real_tab, rand_tab, njacks_ra=1, njacks_dec=30, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', equalArea=True, maskFile=None):

	if(equalArea):
		njacks = njacks_ra*njanks_dec
		jackknife_samples=makeJkSamples_equalArea(real_tab, rand_tab, njacks=njacks, realracol=realracol,realdeccol=realdeccol,randracol=randracol, randdeccol=randdeccol, maskFile=maskFile, plot=True)
	else:
		jackknife_samples=makeJkSamples(real_tab, rand_tab, njacks_ra=njacks_ra, njacks_dec=njacks_dec, realracol=realracol,realdeccol=realdeccol,randracol=randracol, randdeccol=randdeccol)
	
	
	if 0 <= (sample_nr-1) < len(jackknife_samples):
        	return jackknife_samples[sample_nr-1]
	else:
		raise IndexError(f"Sample number {sample_nr} is out of range.")



