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

def omegapRp(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, cosmology_H0_Om0, CFestimator, ra_units='deg', dec_units='deg'):

	H0, Om0=cosmology_H0_Om0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

	dist_real = comovingDistance_Mpch(z_real, cosmology)
	dist_rand = comovingDistance_Mpch(z_rand, cosmology)
	
	H0, OmegaM = cosmology.H0.value, cosmology.Om0

	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=int(rp_nbins), seppmin=rp_min, dsepp=rp_dsep, calcdist=False, logsepp=True, nsepv=int(pi_nbins), dsepv=pi_dsep, estimator=CFestimator, doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.pcf(gals, rans, par, write=False)
	rp = result['rpm']
	omega_p = result['wrp']
	dd = result['dd']

	return rp, omega_p
	
def weightedOmegapRp(ra_real, dec_real, z_real, weight_real, ra_rand, dec_rand, z_rand, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, cosmology_H0_Om0, CFestimator, ra_units='deg', dec_units='deg'):

	H0, Om0=cosmology_H0_Om0
	cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

	dist_real = comovingDistance_Mpch(z_real, cosmology)
	dist_rand = comovingDistance_Mpch(z_rand, cosmology)
	
	H0, OmegaM = cosmology.H0.value, cosmology.Om0

	gals = Table([ra_real, dec_real, z_real, dist_real], names=('ra', 'dec', 'z', 'dcom'))
	rans = Table([ra_rand, dec_rand, z_rand, dist_rand], names=('ra', 'dec', 'z', 'dcom'))
	
	par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=int(rp_nbins), seppmin=rp_min, dsepp=rp_dsep, calcdist=False, logsepp=True, nsepv=int(pi_nbins), dsepv=pi_dsep, estimator=CFestimator, doboot=False) 
	
	gals['wei'] = weight_real/np.mean(weight_real) # gundam does not normalize the weight inside it. 
	rans['wei'] = 1.

	result = gun.pcf(gals, rans, par)
	rp = result['rpm']
	weighted_omega_p = result['wrp']
	ww = result['dd']
	
	return rp, weighted_omega_p
	
def projectedMcf(rp, omega_p, weighted_omega_p):
	Mp_rp = (1 + (weighted_omega_p/rp))/(1 + (omega_p/rp))
	return Mp_rp
	
def computeCF(real_tab, real_properties, rand_tab, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, realracol,realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0, ranked, CFestimator):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]
	z_real = real_tab[realzcol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	z_rand = rand_tab[randzcol]
	
	rp, omega_p = omegapRp(ra_real, dec_real, z_real, ra_rand, dec_rand, z_rand, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, cosmology_H0_Om0, CFestimator)
	
	rp_omega_mcfs = np.empty((len(rp), 0))
	
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, rp.reshape(len(rp), 1)))
	rp_omega_mcfs = np.hstack((rp_omega_mcfs, omega_p.reshape(len(rp), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		if(ranked):
			weight_real = rankdata(prop_now)
		else:
			weight_real = prop_now 
		
		rp, weighted_omega_ranked = weightedOmegapRp(ra_real, dec_real, z_real, weight_real, ra_rand, dec_rand, z_rand, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, cosmology_H0_Om0, CFestimator)
	
		Mp_rp = np.array(projectedMcf(rp, omega_p, weighted_omega_ranked)).reshape(len(rp), 1)
				
		rp_omega_mcfs = np.hstack((rp_omega_mcfs, Mp_rp))
		
	return rp_omega_mcfs
	
def runComputationProjected(real_tab, real_properties, rand_tab, rp_min=0.01, rp_max=100.0, rp_nbins=None, rp_dsep=None, pi_min=0.0, pi_max=40.0, pi_nbins=None, pi_dsep=None, njacks_ra=0, njacks_dec=0, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC', realzcol='z', randracol='RA', randdeccol='Dec', randzcol='z', omp=False, cosmology_H0_Om0=[70.0, 0.3], ranked=True, CFestimator='LS'):

    # setting bins in rp
    if rp_nbins is not None and rp_dsep is not None:
        rp_dsep = (np.log10(rp_max) - np.log10(rp_min)) / rp_nbins
    elif rp_nbins is None and rp_dsep is None:
        rp_nbins = 10 # default 10 bins in rp
        rp_dsep = (np.log10(rp_max) - np.log10(rp_min)) / rp_nbins
    elif rp_nbins is None:
        rp_nbins = int((np.log10(rp_max) - np.log10(rp_min)) / rp_dsep)
    elif rp_dsep is None:
        rp_dsep = (np.log10(rp_max) - np.log10(rp_min)) / rp_nbins

    # setting bins in pi
    if pi_nbins is not None and pi_dsep is not None:
        pi_dsep = (pi_max-pi_min)/pi_nbins
    elif pi_nbins is None and pi_dsep is None:
        pi_nbins = int(40) # default 40 bins in pi
        pi_dsep = (pi_max-pi_min)/pi_nbins
    elif pi_nbins is None:
        pi_nbins = int((pi_max-pi_min)/pi_dsep)
    elif pi_dsep is None:
        pi_dsep = (pi_max-pi_min)/pi_nbins
        
    print(rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep)
	
    os.chdir(working_dir)
    os.mkdir('biproducts')
    os.mkdir('results')
    os.mkdir('results/jackknifes')

    def process_jackknife(jk_i):
    
        if(jk_i == 0):
            real_tab_i, rand_tab_i = real_tab, rand_tab 
            result_file = 'results/CFReal.txt'
            print("Working on the real sample")
        else:
            real_tab_i, rand_tab_i = jackknife_samples[jk_i - 1]
            result_file = 'results/jackknifes/CFJackknife_jk%d.txt' %jk_i
            print("Working on the jackknife sample %d" %jk_i)
			
        result_i = computeCF(real_tab_i, real_properties, rand_tab_i, rp_min, rp_nbins, rp_dsep, pi_nbins, pi_dsep, realracol, realdeccol, realzcol, randracol, randdeccol, randzcol, cosmology_H0_Om0, ranked, CFestimator)
        
        np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
        
        return None
	
    n_jacks = njacks_ra * njacks_dec
	
    jackknife_samples = jkgen.makeJkSamples(real_tab, rand_tab, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)

    if(omp): #TODO: not working...
        logging.info("Parallel programming with %d workers...", os.cpu_count())
        with ProcessPoolExecutor() as executor:
            executor.map(process_jackknife, range(n_jacks + 1))
    else:
        for jk_i in range(n_jacks+1):
            process_jackknife(jk_i)
	
    return None
	

