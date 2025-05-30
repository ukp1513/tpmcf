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

def comovingDistanceH0(redshift, cosmology):
	little_h = cosmology.H0.value/100.
	comDist = cosmology.comoving_distance(redshift).value*little_h
	return comDist
	
def xiS(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, nbins, cosmology, bin_type='Log', ra_units='deg', dec_units='deg'):

	dist_real = comovingDistanceH0(z_real, cosmology)
	dist_rand = comovingDistanceH0(z_rand, cosmology)

	H0, OmegaM = cosmology.H0.value, cosmology.Om0

	log_bin_width = (np.log10(s_max) - np.log10(s_min)) / (nbins)
	
	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='rcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nseps=nbins, sepsmin=s_min, dseps=log_bin_width, calcdist=False, logseps=True, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.rcf(gals, rans, par)
	s = result['sm']
	xi_s = result['xis']

	return s, xi_s
	
def weightedXiS(ra_real, dec_real, z_real, weight_real, ra_rand, dec_rand, z_rand, s_min, s_max, nbins, cosmology, bin_type='Log', ra_units='deg', dec_units='deg'):

	dist_real = comovingDistanceH0(z_real, cosmology)
	dist_rand = comovingDistanceH0(z_rand, cosmology)
	
	H0, OmegaM = cosmology.H0.value, cosmology.Om0
	
	log_bin_width = (np.log10(s_max) - np.log10(s_min)) / (nbins)
	
	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='rcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nseps=nbins, sepsmin=s_min, dseps=log_bin_width, calcdist=False, logseps=True, estimator='LS', doboot=False) 
	
	gals['wei'] = weight_real
	rans['wei'] = 1.

	result = gun.rcf(gals, rans, par)
	s = result['sm']
	weighted_xi_s = result['xis']

	return s, weighted_xi_s
	
def mcfS(s, xi_s, weighted_xi_s):
	M_s = (1 + weighted_xi_s)/(1 + xi_s)
	return M_s
	
def computeCF(real_tab, real_properties, rand_tab, s_min, s_max, nbins, bin_type, ranked, ra_units, dec_units, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	s, xi = xiS(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, s_min, s_max, nbins, cosmology=cosmology)
	
	s_xi_mcfs = np.empty((len(s), 0))
	
	s_xi_mcfs = np.hstack((s_xi_mcfs, s.reshape(len(s), 1)))
	s_xi_mcfs = np.hstack((s_xi_mcfs, xi.reshape(len(s), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
	
		if(ranked == True):
			prop_now_ranked = rankdata(prop_now)
			weight_real = prop_now_ranked
		else:
			weight_real = prop_now
			
		s, weighted_xi_ranked = weightedXiS(ra_real, dec_real, z_real, weight_real, ra_rand, dec_rand, z_rand, s_min, s_max, nbins, cosmology=cosmology)
		
		M_s = np.array(mcfS(s, xi, weighted_xi_ranked)).reshape(len(s), 1)
				
		s_xi_mcfs = np.hstack((s_xi_mcfs, M_s))
		
		
	return s_xi_mcfs
	
	
def runComputation3D(real_tab, real_properties, rand_tab, njacks_ra, njacks_dec, working_dir=os.getcwd(), s_min=5.0, s_max=5000.0, nbins=8, bin_type='Log', ranked=True, ra_units='deg', dec_units='deg', realracol='RA',realdeccol='DEC', realzcol='redshift', randracol='RA', randdeccol='Dec', randzcol='redshift', cosmology_H0_Om0=[70.0, 0.3]):

	H0, Om0=cosmology_H0_Om0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
	
	os.chdir(working_dir)
	if(os.path.exists('biproducts')):
		shutil.rmtree('biproducts')
	os.mkdir('biproducts')
	
	if(os.path.exists('results')):
		shutil.rmtree('results')
	os.mkdir('results')
	
	if(os.path.exists('results/jackknifes')):
		shutil.rmtree('results/jackknifes')
	os.mkdir('results/jackknifes')
	
	global realGal, randGal
	realGal = real_tab
	randGal = rand_tab
	
	n_jacks = njacks_ra * njacks_dec
	
	print('Nr. of jackknifes: ', n_jacks)

	for jk_i in range(n_jacks+1):
		if(jk_i == 0):
			real_tab_i, rand_tab_i = real_tab, rand_tab 
			result_file = 'results/CFReal.txt'
			print("Working on the real sample")
		else:
			real_tab_i, rand_tab_i = jkgen.giveJkSample(jk_i, real_tab, rand_tab, njacks_ra=njacks_ra, njacks_dec=njacks_dec, realracol=realracol, realdeccol=realdeccol, randracol=randracol, randdeccol=randdeccol)
			result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
			print("Working on the jackknife sample %d" %jk_i)
			
		result_i = computeCF(real_tab_i, real_properties, rand_tab_i, s_min, s_max, nbins, bin_type, ranked, ra_units, dec_units, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology=cosmology)
		
		np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
	
	return None

