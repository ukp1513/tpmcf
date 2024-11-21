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
from . import jkgen
import gundam as gun

logging.basicConfig(level=logging.INFO)

def omegaTheta(real_tab, rand_tab, th_min=0.001, nbins=8, d_th=0.3, ra_units='deg', dec_units='deg', sep_units='degrees', realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	

	par = gun.packpars(kind='acf', nsept=nbins, septmin=th_min, dsept=d_th, logsept=True, cra=realracol, cdec=realdeccol, cra1=randracol,cdec1=randdeccol, estimator='LS', doboot=False) 
	
	real_tab['wei'] = 1.
	rand_tab['wei'] = 1.

	result = gun.acf(real_tab, rand_tab, par)
	th = result['thm']
	omega = result['wth']

	return th, omega
	
def weightedOmegaTheta(real_tab, rand_tab, weight_real, th_min=0.001, nbins=8, d_th=0.3, ra_units='deg', dec_units='deg', sep_units='degrees', realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	par = gun.packpars(kind='acf', nsept=nbins, septmin=th_min, dsept=d_th, logsept=True, cra=realracol, cdec=realdeccol, cra1=randracol,cdec1=randdeccol, estimator='LS', doboot=False) 
	
	real_tab['wei'] = weight_real
	rand_tab['wei'] = 1.

	result = gun.acf(real_tab, rand_tab, par)
	th = result['thm']
	weighted_omega = result['wth']

	return th, weighted_omega
	
def mcfTheta(th, omega_th, weighted_omega_th):
	M_th = (1 + weighted_omega_th)/(1 + omega_th)
	return M_th
	
def computeCF(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	
	d_th = (np.log10(th_max) - np.log10(th_min)) / nbins

	th, omega = omegaTheta(real_tab, rand_tab, th_min=thmin, th_max=thmax, d_th=d_th, realracol=realracol,realdeccol=realdeccol,randracol=randracol, randdeccol=randdeccol)
	
	th_omega_mcfs = np.empty((len(th), 0))
	
	th_omega_mcfs = np.hstack((th_omega_mcfs, th.reshape(len(th), 1)))
	th_omega_mcfs = np.hstack((th_omega_mcfs, omega.reshape(len(th), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
		
		th, weighted_omega_ranked = weightedOmegaTheta(real_tab, rand_tab, weight_real=prop_now_ranked, th_min=thmin, th_max=thmax, d_th=d_th, realracol=realracol,realdeccol=realdeccol,randracol=randracol, randdeccol=randdeccol)
	
		th, weighted_omega_ranked = weightedOmegaTheta(ra_real, dec_real, prop_now_ranked, ra_rand, dec_rand, th_min=thmin, th_max=thmax, nbins=th_nbins)

		M_theta = np.array(mcfTheta(th, omega, weighted_omega_ranked)).reshape(len(th), 1)
				
		th_omega_mcfs = np.hstack((th_omega_mcfs, M_theta))
		
	return th_omega_mcfs
	

	
def runComputationAngular(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

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
	

