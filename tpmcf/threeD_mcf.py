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
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed

from . import jkgen



def comovingDistanceH0(redshift, cosmology):
	comDist = cosmology.comoving_distance(redshift)*cosmology.H0/100.
	return comDist
	
def xiS(ra_real, dec_real, dist_real, ra_rand, dec_rand, dist_rand, s_min, s_max, bin_size, bin_type, ra_units, dec_units):

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
	
def weightedXiS(ra_real, dec_real, dist_real, weight_real, ra_rand, dec_rand, dist_rand, s_min, s_max, bin_size, bin_type, ra_units, dec_units):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, w=weight_real, r=dist_real, ra_units=ra_units, dec_units=dec_units)
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
	weighted_xi, var_weightedxi = ww.calculateXi(rr=rr, dr=wr)
	s = np.exp(ww.meanlogr)

	return s, weighted_xi
	
def mcfS(s, xi_s, weighted_xi_s):
	M_s = (1 + weighted_xi_s)/(1 + xi_s)
	return M_s
	
def computeCF(real_tab, real_properties, rand_tab, s_min, s_max, bin_size, bin_type, ranked, ra_units, dec_units, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	dist_real = comovingDistanceH0(z_real, cosmology)
	dist_rand = comovingDistanceH0(z_rand, cosmology)

	s, xi = xiS(ra_real, dec_real, dist_real, ra_rand, dec_rand, dist_rand, s_min, s_max, bin_size, bin_type, ra_units, dec_units)
	
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
			
		s, weighted_xi_ranked = weightedXiS(ra_real, dec_real, dist_real, weight_real, ra_rand, dec_rand, dist_rand, s_min, s_max, bin_size, bin_type, ra_units, dec_units)

		M_s = np.array(mcfS(s, xi, weighted_xi_ranked)).reshape(len(s), 1)
				
		s_xi_mcfs = np.hstack((s_xi_mcfs, M_s))
		
		
	return s_xi_mcfs
	
	
def runComputation3D(real_tab, real_properties, rand_tab, njacks_ra, njacks_dec, working_dir=os.getcwd(), s_min=5.0, s_max=5000.0, bin_size=0.5, bin_type='Log', ranked=True, ra_units='deg', dec_units='deg', realracol='RA',realdeccol='DEC', realzcol='redshift', randracol='RA', randdeccol='Dec', randzcol='redshift', cosmology_H0_Om0=[70.0, 0.3]):

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
	
	for jk_i in range(n_jacks+1):
		if(jk_i == 0):
			real_tab_i, rand_tab_i = real_tab, rand_tab 
			result_file = 'results/CFReal.txt'
			print("Working on the real sample")
		else:
			real_tab_i, rand_tab_i = jkgen.giveJkSample(jk_i, real_tab, rand_tab, njacks_ra=njacks_ra, njacks_dec=njacks_dec, realracol=realracol, realdeccol=realdeccol, randracol=randracol, randdeccol=randdeccol)
			result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
			print("Working on the jackknife sample %d" %jk_i)
			
		result_i = computeCF(real_tab_i, real_properties, rand_tab_i, s_min, s_max, bin_size, bin_type, ranked, ra_units, dec_units, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology=cosmology)
		
		np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
	
	return None

