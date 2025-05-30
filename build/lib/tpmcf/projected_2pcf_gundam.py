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

def comovingDistance_Mpch(redshift, cosmology):
	little_h = cosmology.H0.value/100.
	comDist = cosmology.comoving_distance(redshift).value*little_h
	return comDist
	
def omegapRp(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, rp_min, rp_max, rp_nbins, pi_min, pi_max, cosmology_H0_Om0, ra_units='deg', dec_units='deg'):

	H0, Om0=cosmology_H0_Om0
	little_h = H0/100.0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
	
	dist_real = comovingDistance_Mpch(z_real, cosmology)
	dist_rand = comovingDistance_Mpch(z_rand, cosmology)
	
	pi_max_possible = max(dist_real)-min(dist_real)
	print(pi_max_possible)
	
	
	if(pi_max > pi_max_possible):
		pi_nbins = int(np.floor(pi_max_possible))
		print("Set pi_max %0.2f Mpc/h is larger than the comoving radial separation of sample - setting to %0.2f Mpc/h..." %(pi_max, float(pi_nbins)))
	else:
		pi_nbins = int(np.floor(pi_max))
		 
	
	log_d_rp = (np.log10(rp_max) - np.log10(rp_min)) / (rp_nbins)
	
	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='pcf', h0=H0, omegam=Om0, omegal=(1-Om0), nsepp=rp_nbins, seppmin=rp_min, dsepp=log_d_rp, calcdist=False, logsepp=True, nsepv=pi_nbins, dsepv=1.0, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.pcf(gals, rans, par, write=False)
	rp = result['rpm']
	omega_p = result['wrp']
	dd = result['dd']
	
	return rp, omega_p
	

def computeCF(real_tab, rand_tab, rpmin, rpmax, rp_nbins, pimin, pimax, realracol,realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0):

	little_h = cosmology_H0_Om0[0]/100.0

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	rp, omega_p = omegapRp(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, rpmin, rpmax, rp_nbins, pimin, pimax, cosmology_H0_Om0)
	
	rp_omegas = np.empty((len(rp), 0))
	
	rp_omegas = np.hstack((rp_omegas, rp.reshape(len(rp), 1)))
	rp_omegas = np.hstack((rp_omegas, omega_p.reshape(len(rp), 1)))
	
	return rp_omegas
	
def runComputationProjected(real_tab, rand_tab, rpmin, rpmax, rp_nbins, pimin, pimax, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC', realzcol='z', randracol='RA', randdeccol='Dec', randzcol='z', omp=False, cosmology_H0_Om0=[70.0, 0.3]):

	
	
	os.chdir(working_dir)
	os.mkdir('biproducts')
	os.mkdir('results')
	os.mkdir('results/jackknifes')
	
	def process_jackknife(jk_i):

		try:
			if(jk_i == 0):
				real_tab_i, rand_tab_i = real_tab, rand_tab 
				result_file = 'results/CFReal.txt'
				print("Working on the real sample (Nreal = %d, Nrandom = %d)" %(len(real_tab_i), len(rand_tab_i)))
			else:
				real_tab_i, rand_tab_i = jackknife_samples[jk_i - 1]
				result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
				print("Working on the jackknife sample %d (Nreal = %d, Nrandom = %d)" %(jk_i, len(real_tab_i), len(rand_tab_i)))
			
	
			result_i = computeCF(real_tab_i, rand_tab_i, rpmin, rpmax, rp_nbins, pimin, pimax, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0)
			
			np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
			
		except Exception as e:
			logging.error("Error processing jk_i = %d: %s", jk_i, e)
			return 1
			
		return 0
	
	n_jacks = njacks_ra * njacks_dec
	
	jackknife_samples = jkgen.makeJkSamples(real_tab, rand_tab, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)
	
	if(omp): #TODO: not working...
		logging.info("Parallel programming with %d workers...", os.cpu_count())
		with ProcessPoolExecutor() as executor:
			executor.map(process_jackknife, range(n_jacks + 1))
	else:
		for jk_i in range(n_jacks+1):
			process_outcome = process_jackknife(jk_i)
	
	return process_outcome
	

