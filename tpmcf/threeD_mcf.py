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



def comovingDistanceH0(redshift, cosmology):
	comDist = cosmology.comoving_distance(redshift)*cosmology.H0/100.
	return comDist
	
def xiS(ra_real, dec_real, dist_real, ra_rand, dec_rand, dist_rand, s_min=5.0, s_max=5000.0, bin_size=10.0, bin_type='Linear', ra_units='deg', dec_units='deg'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, r=dist_real, ra_units=ra_units, dec_units=dec_units)
	dd = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	dd.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, r=dist_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	rr.process(cat_rand)

	# Create their cross catalog
	dr = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	dr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	xi, varxi = dd.calculateXi(rr=rr, dr=dr)
	s = np.exp(dd.meanlogr)

	return s, xi
	
def weightedXiS(ra_real, dec_real, dist_real, weight_real, ra_rand, dec_rand, dist_rand, s_min=5.0, s_max=5000.0, bin_size=10.0, bin_type='Linear', ra_units='deg', dec_units='deg'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, w=weighted_real, r=dist_real, ra_units=ra_units, dec_units=dec_units)
	ww = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	ww.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, r=dist_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	rr.process(cat_rand)

	# Create their cross catalog
	wr = treecorr.NNCorrelation(bin_type=bin_type, min_sep=s_min, max_sep=s_max, bin_size=bin_size)
	wr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	weighted_xi, var_weightedxi = dd.calculateXi(rr=rr, dr=dr)
	s = np.exp(dd.meanlogr)

	return s, weighted_xi
	
def mcfS(s, xi_s, weighted_xi_s):
	M_s = (1 + weighted_xi_s)/(1 + xi_s)
	return M_s
	
def computeCF(real_tab, real_properties, rand_tab, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	dist_real = comoving_distance_H0(z_real, cosmology)
	dist_rand = comoving_distance_H0(z_rand, cosmology)

	s, xi = xiS(ra_real, dec_real, dist_real, ra_rand, dec_rand, dist_rand)
	
	s_xi_mcfs = np.empty((len(s), 0))
	
	s_xi_mcfs = np.hstack((s_xi_mcfs, s.reshape(len(s), 1)))
	s_xi_mcfs = np.hstack((s_xi_mcfs, xi.reshape(len(s), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
	
		s, weighted_xi_ranked = weightedXiS(ra_real, dec_real, dist_real, prop_now_ranked, ra_rand, dec_rand, dist_rand)

		M_s = np.array(mcfS(s, xi, weighted_xi_ranked)).reshape(len(s), 1)
				
		s_xi_mcfs = np.hstack((s_xi_mcfs, M_s))
		
	return s_xi_mcfs
	
	
def runComputation(real_tab, real_properties, rand_tab, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC', realzcol='redshift', randracol='RA', randdeccol='Dec', randzcol='redshift', cosmology_H0_Om0=[70.0, 0.3]):

	H0, Om0=cosmology_H0_Om0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

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
			
		result_i = computeCF(real_tab_i, real_properties, rand_tab_i, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology=cosmology)
		
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
	

