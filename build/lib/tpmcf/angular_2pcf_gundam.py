import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata
from astropy.table import Table
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
from . import jkgen
import gundam as gun

logging.basicConfig(level=logging.INFO)

def omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, nbins=8, ra_units='deg', dec_units='deg', sep_units='degrees'):
	
	log_bin_width = (np.log10(th_max) - np.log10(th_min)) / (nbins)
	
	gals = Table([ra_real, dec_real], names=('ra', 'dec'))
	rans = Table([ra_rand, dec_rand], names=('ra', 'dec'))	
	
	par = gun.packpars(kind='acf', nsept=nbins, septmin=th_min, dsept=log_bin_width, logsept=True, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.acf(gals, rans, par)
	th = result['thm']
	omega = result['wth']

	return th, omega
	

	
def computeCF(real_tab, rand_tab, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	
	d_th = (np.log10(thmax) - np.log10(thmin)) / th_nbins

	th, omega = omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=thmin, th_max=thmax, nbins=th_nbins)
	
	th_omegas = np.empty((len(th), 0))
	
	th_omegas = np.hstack((th_omegas, th.reshape(len(th), 1)))
	th_omegas = np.hstack((th_omegas, omega.reshape(len(th), 1)))
	
		
	return th_omegas
	

	
def runComputationAngular(real_tab, rand_tab, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

	os.chdir(working_dir)
	os.mkdir('biproducts')
	os.mkdir('results')
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
			
			result_i = computeCF(real_tab_i, rand_tab_i, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol)
			
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
	

