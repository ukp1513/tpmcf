import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata

from astropy.cosmology import FlatLambdaCDM

import treecorr
import healpy as hp
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

from . import jkgen


def omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, bin_size=0.5, ra_units='deg', dec_units='deg', sep_units='degrees'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, ra_units=ra_units, dec_units=dec_units)
	dd = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	dd.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	rr.process(cat_rand)

	# Create their cross catalog
	dr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	dr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	omega, varomega = dd.calculateXi(rr=rr, dr=dr)
	th = np.exp(dd.meanlogr)

	return th, omega
	
def weightedOmegaTheta(ra_real, dec_real, weight_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, bin_size=0.5, ra_units='deg', dec_units='deg', sep_units='degrees'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, w=weight_real, ra_units=ra_units, dec_units=dec_units)
	ww = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	ww.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	rr.process(cat_rand)

	# Create their cross catalog
	wr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, bin_size=bin_size, sep_units = sep_units)
	wr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	weighted_omega, var_weightedomega = ww.calculateXi(rr=rr, dr=wr)
	th = np.exp(ww.meanlogr)

	return th, weighted_omega
	

def mcfTheta(th, omega_th, weighted_omega_th):
	M_th = (1 + weighted_omega_th)/(1 + omega_th)
	return M_th
	
def computeCF(real_tab, real_properties, rand_tab, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]

	th, omega = omegaTheta(ra_real, dec_real, ra_rand, dec_rand)
	
	th_omega_mcfs = np.empty((len(th), 0))
	
	th_omega_mcfs = np.hstack((th_omega_mcfs, th.reshape(len(th), 1)))
	th_omega_mcfs = np.hstack((th_omega_mcfs, omega.reshape(len(th), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
	
		th, weighted_omega_ranked = weightedOmegaTheta(ra_real, dec_real, prop_now_ranked, ra_rand, dec_rand)

		M_theta = np.array(mcfTheta(th, omega, weighted_omega_ranked)).reshape(len(th), 1)
				
		th_omega_mcfs = np.hstack((th_omega_mcfs, M_theta))
		
	return th_omega_mcfs
	
	
def runComputation(real_tab, real_properties, rand_tab, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	os.chdir(working_dir)
	os.mkdir('biproducts')
	os.mkdir('results')
	os.mkdir('results/jackknifes')
	
	global realGal, randGal
	realGal = real_tab
	randGal = rand_tab
	
	n_jacks = njacks_ra * njacks_dec
	
	for jk_i in range(n_jacks+1):
		if(jk_i == 0):
			real_tab_i, rand_tab_i = real_tab, rand_tab 
			result_file = 'results/CFReal.txt'
		else:
			real_tab_i, rand_tab_i = jkgen.giveJkSample(jk_i, real_tab, rand_tab, njacks_ra=njacks_ra, njacks_dec=njacks_dec, realracol=realracol, realdeccol=realdeccol, randracol=randracol, randdeccol=randdeccol)
			result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
			
		result_i = computeCF(real_tab_i, real_properties, rand_tab_i, realracol, realdeccol, randracol, randdeccol)
		
		np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
	
	'''
	
	with ProcessPoolExecutor() as executor:
		futures = []
		for i in range(n_jacks + 1):
			result_file = 'results/CFReal.txt' if i == 0 else f'results/jackknifes/CFJackknife_jk{i}.txt'

			# Parallelize the computation directly in the loop
			futures.append(executor.submit(computeCF, real_tab, real_properties, rand_tab, realracol, realdeccol, randracol, randdeccol))
	
	for future, i in zip(as_completed(futures), range(n_jacks + 1)):
		try:
			result_i = future.result()  # Get the computed result
			result_file = 'results/CFReal.txt' if i == 0 else f'results/jackknifes/CFJackknife_jk{i}.txt'
			np.savetxt(result_file, result_i, delimiter="\t", fmt='%f')  # Save result
		except Exception as exc:
			print(f"Sample computation generated an exception: {exc}")
	'''
	return None

	##################
	'''
	
	for i in range(n_jacks+1):
		if(i == 0):
			result_file = 'results/CFReal.txt'
		else:
			result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %i
			
		result_i = computeSample(i)
		np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
	
	
	# Use ProcessPoolExecutor for parallel computation
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(compute_sample, i) for i in range(n_jacks + 1)]

	# Optionally, wait for all tasks to complete and handle any errors
	for future in as_completed(futures):
		try:
			future.result()
		except Exception as exc:
			print(f"Sample computation generated an exception: {exc}")
	

	'''
	
