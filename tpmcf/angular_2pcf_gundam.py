import numpy as np 
from astropy.table import Table
import os
import logging
from . import jkgen
import gundam as gun
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO)

def omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=0.001, th_max=50.0, nbins=8, ra_units='deg', dec_units='deg', sep_units='degrees'):
	
	log_bin_width = (np.log10(th_max) - np.log10(th_min)) / (nbins)
	
	gals = Table([ra_real, dec_real], names=('ra', 'dec'))
	rans = Table([ra_rand, dec_rand], names=('ra', 'dec'))	
	
	par = gun.packpars(kind='acf', nsept=nbins, septmin=th_min, dsept=log_bin_width, logsept=True, estimator='LS', doboot=False) 
	
	gals['wei'] = 1.
	rans['wei'] = 1.

	result = gun.acf(gals, rans, par)
	th = result['thm']
	omega = result['wth']

	return th, omega
	
def computeCF(real_tab, rand_tab, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]
	
	d_th = (np.log10(thmax) - np.log10(thmin)) / th_nbins

	th, omega = omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min=thmin, th_max=thmax, nbins=th_nbins)
	
	th_omegas = np.empty((len(th), 0))
	
	th_omegas = np.hstack((th_omegas, th.reshape(len(th), 1)))
	th_omegas = np.hstack((th_omegas, omega.reshape(len(th), 1)))
		
	return th_omegas
	
def _process_jackknife(args):

    jk_i, real_tab_arg, rand_tab_arg, thmin_arg, thmax_arg, th_nbins_arg, realracol_arg, realdeccol_arg, randracol_arg, randdeccol_arg, jackknife_samples_arg, working_dir = args

    try:
        if(jk_i == 0):
            real_tab_i, rand_tab_i = real_tab_arg, rand_tab_arg 
            result_file = os.path.join(working_dir, 'results', 'CFReal.txt')
            print("Working on the real sample: Nreal = %d, Nrand = %d" %(len(real_tab_i), len(rand_tab_i)))
        else:
            real_tab_i, rand_tab_i = jackknife_samples_arg[jk_i - 1]
            result_file = os.path.join(working_dir, 'results', 'jackknifes', 'CFJackknife_jk%d.txt' %jk_i)
            print("Working on the jackknife sample %d: Nreal = %d, Nrand = %d" %(jk_i, len(real_tab_i), len(rand_tab_i)))
		
        result_i = computeCF(real_tab_i, rand_tab_i, thmin_arg, thmax_arg, th_nbins_arg, realracol_arg, realdeccol_arg, randracol_arg, randdeccol_arg)

        np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
		
    except Exception as e:
        logging.error("Error processing jk_i = %d: %s", jk_i, e)
        return 1

    return 0
    
def runComputationAngular(real_tab, rand_tab, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

    os.chdir(working_dir)
    os.makedirs(working_dir+os.path.sep+'biproducts',  exist_ok=True)
    os.makedirs(working_dir+os.path.sep+'results/jackknifes',  exist_ok=True)

    jackknife_samples = jkgen.makeJkSamples(real_tab, rand_tab, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)

    n_jacks = njacks_ra * njacks_dec

    process_outcomes = []
    
    if(omp): 
        num_processes = cpu_count()
        print(f"Parallelizing with {num_processes} processes...")
        
        tasks = []
        for jk_i in range(n_jacks + 1):
            tasks.append((jk_i, real_tab, rand_tab, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol, jackknife_samples, working_dir))
	
        with Pool(processes=num_processes) as pool:
            process_outcomes = pool.map(_process_jackknife, tasks)
    else:
        for jk_i in range(n_jacks+1):
            args = (jk_i, real_tab, rand_tab, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol, jackknife_samples, working_dir)
            outcome = _process_jackknife(args)
            process_outcomes.append(outcome)
	
    if any(outcome != 0 for outcome in process_outcomes):
        print("Warning: Some jackknife computations failed.")
        return 1
    else:
        print("All computations completed successfully.")
        return 0

