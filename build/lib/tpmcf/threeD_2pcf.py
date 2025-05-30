import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import gundam as gun
import treecorr
import healpy as hp
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed

from . import jkgen
import logging
logging.basicConfig(level=logging.INFO)

def comovingDistanceH0(redshift, cosmology):
	little_h = cosmology.H0.value/100.
	comDist = cosmology.comoving_distance(redshift).value*little_h
	return comDist
	
def xiS_treecorr(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, s_nbins, cosmology_H0_Om0, bin_type='Log', ra_units='deg', dec_units='deg'):

	H0, Om0=cosmology_H0_Om0
	little_h = H0/100.0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

	dist_real = comovingDistanceH0(z_real, cosmology)
	dist_rand = comovingDistanceH0(z_rand, cosmology)
	
	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, r=dist_real, ra_units=ra_units, dec_units=dec_units)
	dd = treecorr.NNCorrelation(min_sep=s_min, max_sep=s_max, nbins=s_nbins, bin_type=bin_type)
	dd.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, r=dist_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=s_min, max_sep=s_max, nbins=s_nbins, bin_type=bin_type)
	rr.process(cat_rand)

	# Create their cross catalog
	dr = treecorr.NNCorrelation(min_sep=s_min, max_sep=s_max, nbins=s_nbins, bin_type=bin_type)
	dr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	xi, varxi = dd.calculateXi(rr=rr, dr=dr)
	s = np.exp(dd.meanlogr)

	return s, xi
	
def xiS_gundam(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, s_nbins, cosmology_H0_Om0, bin_type='Log', ra_units='deg', dec_units='deg'):

	H0, Om0=cosmology_H0_Om0
	little_h = H0/100.0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

	dist_real = comovingDistanceH0(z_real, cosmology)
	dist_rand = comovingDistanceH0(z_rand, cosmology)

	H0, OmegaM = cosmology.H0.value, cosmology.Om0

	log_bin_width = (np.log10(s_max) - np.log10(s_min)) / (s_nbins)
	
	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='rcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nseps=s_nbins, sepsmin=s_min, dseps=log_bin_width, calcdist=False, logseps=True, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.rcf(gals, rans, par)
	s = result['sm']
	xi_s = result['xis']

	return s, xi_s
	

def mcfS(s, xi_s, weighted_xi_s):
	M_s = (1 + weighted_xi_s)/(1 + xi_s)
	return M_s
	
def computeCF(real_tab, rand_tab, s_min, s_max, s_nbins, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0, code):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	if(code == 'gundam'):
		s, xi = xiS_gundam(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, s_nbins, cosmology_H0_Om0)
	elif(code == 'treecorr'):
		s, xi = xiS_treecorr(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, s_nbins, cosmology_H0_Om0)
	
	s_xi_mcfs = np.empty((len(s), 0))
	
	s_xi_mcfs = np.hstack((s_xi_mcfs, s.reshape(len(s), 1)))
	s_xi_mcfs = np.hstack((s_xi_mcfs, xi.reshape(len(s), 1)))
	
	return s_xi_mcfs
	
	

	
def runComputation3D(real_tab, rand_tab, s_min, s_max, s_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC', realzcol='redshift', randracol='RA', randdeccol='Dec', randzcol='redshift', omp=False, cosmology_H0_Om0=[70.0, 0.3], code='treecorr'):

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
			
	
			result_i = computeCF(real_tab_i, rand_tab_i, s_min, s_max, s_nbins, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0, code)
			
			
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

