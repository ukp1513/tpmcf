from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from astropy.stats import bootstrap
import datetime
import os
from random import sample 
import pandas as pd


def makeJkSamples(real_table, rand_table, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False):

	n_d = len(real_table)

	n_r = len(rand_table)

	jack_nr = 0

	ra_min, ra_max = min(rand_table[randracol]), max(rand_table[randracol])
	dec_min, dec_max = min(rand_table[randdeccol]),max(rand_table[randdeccol])


	# CREATING JACKKNIFE SAMPLES
	jackknife_samples = []

	for jk_r in range(njacks_ra):
		for jk_c in range(njacks_dec):
			jack_nr +=1 

			raMinNow = ra_min + (jk_r * (ra_max-ra_min)/njacks_ra)
			raMaxNow = raMinNow + (ra_max-ra_min)/njacks_ra

			decMinNow = dec_min + (jk_c * (dec_max-dec_min)/njacks_dec)
			decMaxNow = decMinNow + (dec_max-dec_min)/njacks_dec

			real_jk= real_table[((real_table[realracol] <= raMinNow) | (real_table[realracol] >= raMaxNow)) | ((real_table[realdeccol] <= decMinNow) | (real_table[realdeccol] >= decMaxNow))]
			rand_jk= rand_table[((rand_table[randracol] <= raMinNow) | (rand_table[randracol] >= raMaxNow)) | ((rand_table[randdeccol] <= decMinNow) | (rand_table[randdeccol] >= decMaxNow))]
			if(plot):
				plt.scatter(real_jk[realracol], real_jk[realdeccol], s=1, color='blue')
				plt.show()

			jackknife_samples.append((real_jk, rand_jk))
	
	return jackknife_samples
	
def giveJkSample(sample_nr, real_tab, rand_tab, njacks_ra=1, njacks_dec=30, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	jackknife_samples=makeJkSamples(real_tab, rand_tab, njacks_ra=njacks_ra, njacks_dec=njacks_dec, realracol=realracol,realdeccol=realdeccol,randracol=randracol, randdeccol=randdeccol)
	
	if 0 <= (sample_nr-1) < len(jackknife_samples):
        	return jackknife_samples[sample_nr-1]
	else:
		raise IndexError(f"Sample number {sample_nr} is out of range.")



