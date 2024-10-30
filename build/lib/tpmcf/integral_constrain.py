import numpy as np
import treecorr
from astropy.io import ascii

def w_theta_model(theta, A, gamma):
	return A*pow(theta, 1-gamma)

def computeICAngular(randgalaxies, A, gamma, randracol='RA', randdeccol='Dec'):

	ra_rand = randgalaxies[randracol]
	dec_rand = randgalaxies[randdeccol]

	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')
	rr = treecorr.NNCorrelation(bin_type = "Linear", min_sep=0, max_sep=10, nbins=50, sep_units = 'degrees')
	rr.process(cat_rand)
	
	rr_values = rr.npairs
	theta_bins = rr.meanr
	
	num = 0
	den = 0
	for thetai, theta in enumerate(theta_bins):
		omega_i = w_theta_model(theta, A, gamma)
		num += (rr_values[thetai]*omega_i)
		den += rr_values[thetai]
	IC = num/den
	return IC
	
