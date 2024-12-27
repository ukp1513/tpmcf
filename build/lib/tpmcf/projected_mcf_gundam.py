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



def omegap_rp(real_tab, rand_tab, rp_min=0.001, nbins=8, d_rp=0.3, pi_max=40.0, d_pi = 1.0, ra_units='deg', dec_units='deg', sep_units='degrees', realracol='RA',realdeccol='DEC', realzcol='z',randracol='RA', randdeccol='Dec', randzcol='z', cosmology_H0_Om0=[70.0, 0.3]):

	gals = Table.from_pandas(real_tab)
	rans = Table.from_pandas(rand_tab)
	
	H0, OmegaM=cosmology_H0_Om0
	
	
	
	par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=nbins, seppmin=rp_min, dsepp=d_rp, logsepp=True, nsepv=int(pi_max), dsepv=d_pi, cra=realracol, cdec=realdeccol, cred=realzcol, cra1=randracol,cdec1=randdeccol, cred1=randzcol, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.pcf(gals, rans, par)
	rp = result['rpm']
	omega_p = result['wrp']

	return rp, omega_p
	
def weightedOmegap_rp(real_tab, rand_tab, weight_real, rp_min=0.001, nbins=8, d_rp=0.3, pi_max=40.0, d_pi = 1.0, ra_units='deg', dec_units='deg', sep_units='degrees', realracol='RA',realdeccol='DEC', realzcol='z',randracol='RA', randdeccol='Dec', randzcol='z', cosmology_H0_Om0=[70.0, 0.3]):

	gals = Table.from_pandas(real_tab)
	rans = Table.from_pandas(rand_tab)
	
	H0, OmegaM=cosmology_H0_Om0
	
	par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=nbins, seppmin=rp_min, dsepp=d_rp, logsepp=True, nsepv=int(pi_max), dsepv=d_pi, cra=realracol, cdec=realdeccol, cred=realzcol, cra1=randracol,cdec1=randdeccol, cred1=randzcol, estimator='LS', doboot=False) 
	
	gals['wei'] = weight_real
	rans['wei'] = 1.

	result = gun.pcf(gals, rans, par)
	rp = result['rpm']
	weighted_omega_p = result['wrp']

	return rp, weighted_omega_p

def projectedMcf(rp, omega_p, weighted_omega_p):
	Mp_rp = (1 + (weighted_omega_p/rp))/(1 + (omega_p/rp))
	return Mp_rp
	
def computeCF(real_tab, real_properties, rand_tab, rpmin, rpmax, rp_nbins, pimax, pi_nbins, realracol='RA',realdeccol='DEC', realzcol='z',randracol='RA', randdeccol='Dec', randzcol='z', cosmology_H0_Om0=[70.0, 0.3]):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	d_rp = (np.log10(rpmax) - np.log10(rpmin)) / rp_nbins
	d_pi = (pimax - 0.0)/pi_nbins

	rp, omega_p = omegap_rp(real_tab, rand_tab, rp_min=rpmin, nbins=rp_nbins, d_rp=d_rp, pi_max=pimax, d_pi=d_pi, realracol=realracol,realdeccol=realdeccol, realzcol=realzcol, randracol=randracol, randdeccol=randdeccol, randzcol=randzcol, cosmology_H0_Om0=cosmology_H0_Om0)
	
	rp_omega_mcfs = np.empty((len(rp), 0))
	
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, rp.reshape(len(rp), 1)))
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, omega_p.reshape(len(rp), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
		
		rp, weighted_omega_ranked = weightedOmegap_rp(real_tab, rand_tab, weight_real=prop_now_ranked, rp_min=rpmin, nbins=rp_nbins, d_rp=d_rp, pi_max=pimax, d_pi=d_pi, realracol=realracol,realdeccol=realdeccol, realzcol=realzcol, randracol=randracol, randdeccol=randdeccol, randzcol=randzcol, cosmology_H0_Om0=cosmology_H0_Om0)
	
		Mp_rp = np.array(projectedMcf(rp, omega_p, weighted_omega_ranked)).reshape(len(rp), 1)
				
		rp_omega_mcfs = np.hstack((rp_omega_mcfs, Mp_rp))
		
	return rp_omega_mcfs
	
def computeCF_tmp(real_tab, real_properties, rand_tab, rpmin, rpmax, rp_nbins, pimax, pi_nbins, realracol='RA',realdeccol='DEC', realzcol='z',randracol='RA', randdeccol='Dec', randzcol='z', cosmology_H0_Om0=[70.0, 0.3]):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	d_rp = (np.log10(rpmax) - np.log10(rpmin)) / rp_nbins
	d_pi = (pimax - 0.0)/pi_nbins

	rp, omega_p = omegap_rp(real_tab, rand_tab, rp_min=rpmin, nbins=rp_nbins, d_rp=d_rp, pi_max=pimax, d_pi=d_pi, realracol=realracol,realdeccol=realdeccol, realzcol=realzcol, randracol=randracol, randdeccol=randdeccol, randzcol=randzcol, cosmology_H0_Om0=cosmology_H0_Om0)
	
	rp_omega_mcfs = np.empty((len(rp), 0))
	
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, rp.reshape(len(rp), 1)))
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, omega_p.reshape(len(rp), 1)))
	
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
		
		rp, weighted_omega_ranked = weightedOmegap_rp(real_tab, rand_tab, weight_real=prop_now_ranked, rp_min=rpmin, nbins=rp_nbins, d_rp=d_rp, pi_max=pimax, d_pi=d_pi, realracol=realracol,realdeccol=realdeccol, realzcol=realzcol, randracol=randracol, randdeccol=randdeccol, randzcol=randzcol, cosmology_H0_Om0=cosmology_H0_Om0)
	
		Mp_rp = np.array(projectedMcf(rp, omega_p, weighted_omega_ranked)).reshape(len(rp), 1)
				
		#rp_omega_mcfs = np.hstack((rp_omega_mcfs, weighted_omega_ranked.reshape(len(rp), 1)))
		rp_omega_mcfs = np.hstack((rp_omega_mcfs, Mp_rp))
		
	return rp_omega_mcfs
	

	
def runComputationProjected(real_tab, real_properties, rand_tab, rpmin, rpmax, rp_nbins, pimax, pi_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC', realzcol='z', randracol='RA', randdeccol='Dec', randzcol='z', omp=False, cosmology_H0_Om0=[70.0, 0.3]):

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
			
			result_i = computeCF_tmp(real_tab_i, real_properties, rand_tab_i, rpmin, rpmax, rp_nbins, pimax, pi_nbins, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0)
			
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
	

