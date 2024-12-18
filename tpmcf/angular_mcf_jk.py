import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata

from astropy.cosmology import FlatLambdaCDM
import time
import treecorr
import healpy as hp
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

def omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, nbins=8, ra_units='deg', dec_units='deg', sep_units='degrees', njk=30):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, ra_units=ra_units, dec_units=dec_units, npatch=njk)
	dd = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	dd.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units, npatch=njk)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	rr.process(cat_rand)

	# Create their cross catalog
	dr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	dr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	omega, omega_var = dd.calculateXi(rr=rr, dr=dr)
	th = np.exp(dd.meanlogr)
	omega_cov = dd.cov

	return th, omega, omega_cov 
	
def weightedOmegaTheta(ra_real, dec_real, weight_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, nbins=8, ra_units='deg', dec_units='deg', sep_units='degrees', njk=30):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, w=weight_real, ra_units=ra_units, dec_units=dec_units, npatch=njk)
	ww = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	ww.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units, npatch=njk)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	rr.process(cat_rand)

	# Create their cross catalog
	wr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units, var_method='jackknife')
	wr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	weighted_omega, weightedomega_var = ww.calculateXi(rr=rr, dr=wr)
	th = np.exp(ww.meanlogr)
	weightedomega_cov = ww.cov

	return th, weighted_omega, weightedomega_cov

def mcfTheta(th, omega_th, weighted_omega_th, omega_th_err, weighted_omega_th_err):
	A = 1 + weighted_omega_th
	B = 1 + omega_th
	M_th = A / B

	M_th_err = M_th * np.sqrt((weighted_omega_th_err / A)**2 + (omega_th_err / B)**2)

	return M_th, M_th_err
	
def computeCF(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]

	th, omega, omega_cov = omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=thmin, th_max=thmax, nbins=th_nbins)
	
	th_omega_mcfs = np.empty((len(th), 0))
	
	th_omega_mcfs = np.hstack((th_omega_mcfs, th.reshape(len(th), 1)))
	th_omega_mcfs = np.hstack((th_omega_mcfs, omega.reshape(len(th), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
	
		th, weighted_omega_ranked = weightedOmegaTheta(ra_real, dec_real, prop_now_ranked, ra_rand, dec_rand, th_min=thmin, th_max=thmax, nbins=th_nbins)

		M_theta = np.array(mcfTheta(th, omega, weighted_omega_ranked)).reshape(len(th), 1)
				
		th_omega_mcfs = np.hstack((th_omega_mcfs, M_theta))
		
	return th_omega_mcfs
	

	
def runComputationAngular(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

	os.chdir(working_dir)
	os.mkdir('biproducts')
	os.mkdir('finals')
	os.mkdir('results/jackknifes')
	
	def process_jackknife(jk_i):

		try:
			if(jk_i == 0):
				real_tab_i, rand_tab_i = real_tab, rand_tab 
				result_file = 'results/CFReal.txt'
				print("Working on the real sample")
			else:
				real_tab_i, rand_tab_i = jackknife_samples[jk_i - 1]
				result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
				print("Working on the jackknife sample %d" %jk_i)
			
			result_i = computeCF(real_tab_i, real_properties, rand_tab_i, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol)
			
			np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
			
		except Exception as e:
			logging.error("Error processing jk_i = %d: %s", jk_i, e)
			
		return 0
	
	n_jacks = njacks_ra * njacks_dec
	
	jackknife_samples = jkgen.makeJkSamples(real_tab, rand_tab, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)
	
	if(omp): #TODO: not working...
		logging.info("Parallel programming with %d workers...", os.cpu_count())
		with ProcessPoolExecutor() as executor:
			executor.map(process_jackknife, range(n_jacks + 1))
	else:
		for jk_i in range(n_jacks+1):
			process_jackknife(jk_i)
	
	return 0
	

