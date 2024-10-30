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

logging.basicConfig(level=logging.INFO)


	
def omegaTheta_cross(ra_real1, dec_real1, ra_rand1, dec_rand1,ra_real2, dec_real2, ra_rand2, dec_rand2, th_min=0.001, th_max=50.0, nbins=8, ra_units='deg', dec_units='deg', sep_units='degrees'):

	cat_real1 = treecorr.Catalog(ra=ra_real1, dec=dec_real1, ra_units=ra_units, dec_units=dec_units)
	cat_real2 = treecorr.Catalog(ra=ra_real2, dec=dec_real2, ra_units=ra_units, dec_units=dec_units)
	dd = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units)
	dd.process(cat_real1, cat_real2)

	cat_rand1 = treecorr.Catalog(ra=ra_rand1, dec=dec_rand1, ra_units=ra_units, dec_units=dec_units)
	cat_rand2 = treecorr.Catalog(ra=ra_rand2, dec=dec_rand2, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units)
	rr.process(cat_rand1, cat_rand2)

	dr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units)
	dr.process(cat_real1, cat_rand1)

	rd = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, sep_units = sep_units)
	rd.process(cat_real2, cat_rand2)
	
	omega_cross, varomega_cross = dd.calculateXi(rr=rr, dr=dr, rd=rd)
	th = np.exp(dd.meanlogr)

	return th, omega_cross

def computeCF_cross(real_tab1, real_tab2, rand_tab1, rand_tab2, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real1 = real_tab1[realracol]
	dec_real1 = real_tab1[realdeccol]
	ra_rand1 = rand_tab1[randracol]
	dec_rand1 = rand_tab1[randdeccol]
	
	ra_real2 = real_tab2[realracol]
	dec_real2 = real_tab2[realdeccol]
	ra_rand2 = rand_tab2[randracol]
	dec_rand2 = rand_tab2[randdeccol]


	th, omega_cross = omegaTheta_cross(ra_real1, dec_real1, ra_rand1, dec_rand1,ra_real2, dec_real2, ra_rand2, dec_rand2, th_min=thmin, th_max=thmax, nbins=th_nbins)
	
	th_omega_mcfs = np.empty((len(th), 0))
	
	th_omega_mcfs = np.hstack((th_omega_mcfs, th.reshape(len(th), 1)))
	th_omega_mcfs = np.hstack((th_omega_mcfs, omega_cross.reshape(len(th), 1)))
	
	return th_omega_mcfs

	
def runComputationAngular_cross(real_tab1, real_tab2, rand_tab1, rand_tab2, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

	os.chdir(working_dir)
	os.mkdir('biproducts')
	os.mkdir('results')
	os.mkdir('results/jackknifes')
	
	def process_jackknife(jk_i):

		try:
			if(jk_i == 0):
				real_tab_i_1, rand_tab_i_1 = real_tab1, rand_tab1
				real_tab_i_2, rand_tab_i_2 = real_tab2, rand_tab2
				result_file = 'results/CFReal.txt'
				print("Working on the real sample")
			else:
				real_tab_i_1, rand_tab_i_1 = jackknife_samples_1[jk_i - 1]
				real_tab_i_2, rand_tab_i_2 = jackknife_samples_2[jk_i - 1]
				result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
				print("Working on the jackknife sample %d" %jk_i)
			
			result_i = computeCF_cross(real_tab_i_1, real_tab_i_2, rand_tab_i_1, rand_tab_i_2, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol)
			np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
			
		except Exception as e:
			logging.error("Error processing jk_i = %d: %s", jk_i, e)
			
		return 0
	
	n_jacks = njacks_ra * njacks_dec
	
	jackknife_samples_1 = jkgen.makeJkSamples(real_tab1, rand_tab1, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)
	jackknife_samples_2 = jkgen.makeJkSamples(real_tab2, rand_tab2, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)
	
	if(omp): #TODO: not working...
		logging.info("Parallel programming with %d workers...", os.cpu_count())
		with ProcessPoolExecutor() as executor:
			executor.map(process_jackknife, range(n_jacks + 1))
	else:
		for jk_i in range(n_jacks+1):
			process_jackknife(jk_i)
			print("ok")
	
	return 0
	

