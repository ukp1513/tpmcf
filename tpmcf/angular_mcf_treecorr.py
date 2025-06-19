import numpy as np
from scipy.stats import rankdata
import treecorr
import os
import logging
from . import jkgen
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO)

def omegaTheta(ra_real, dec_real, ra_rand, dec_rand, th_min, th_max, nbins, bin_type='Log', ra_units='deg', dec_units='deg', sep_units='degrees'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, ra_units=ra_units, dec_units=dec_units)
	dd = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	dd.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	rr.process(cat_rand)

	# Create their cross catalog
	dr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	dr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	omega, varomega = dd.calculateXi(rr=rr, dr=dr)
	th = np.exp(dd.meanlogr)

	return th, omega
	
def weightedOmegaTheta(ra_real, dec_real, weight_real, ra_rand, dec_rand, th_min, th_max, nbins, bin_type='Log', ra_units='deg', dec_units='deg', sep_units='degrees'):

	# Create catalog for the data
	cat_real = treecorr.Catalog(ra=ra_real, dec=dec_real, w=weight_real, ra_units=ra_units, dec_units=dec_units)
	ww = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	ww.process(cat_real)

	# Create catalog for the randoms
	cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units=ra_units, dec_units=dec_units)
	rr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	rr.process(cat_rand)

	# Create their cross catalog
	wr = treecorr.NNCorrelation(min_sep=th_min, max_sep=th_max, nbins=nbins, bin_type=bin_type, sep_units = sep_units)
	wr.process(cat_real, cat_rand)
	
	# Calculate 2pt correlation function of the total sample
	weighted_omega, var_weightedomega = ww.calculateXi(rr=rr, dr=wr)
	th = np.exp(ww.meanlogr)

	return th, weighted_omega

def mcfTheta(th, omega_th, weighted_omega_th):
	M_th = (1 + weighted_omega_th)/(1 + omega_th)
	return M_th
	
def computeCF(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec'):

	ra_real = real_tab[realracol]
	dec_real = real_tab[realdeccol]

	ra_rand = rand_tab[randracol]
	dec_rand = rand_tab[randdeccol]

	th, omega = omegaTheta(ra_real, dec_real, ra_rand, dec_rand, thmin, thmax, th_nbins)
	
	th_omega_mcfs = np.empty((len(th), 0))
	
	th_omega_mcfs = np.hstack((th_omega_mcfs, th.reshape(len(th), 1)))
	th_omega_mcfs = np.hstack((th_omega_mcfs, omega.reshape(len(th), 1)))
	
	for prop_i in real_properties:
	
		prop_now = np.array(real_tab[prop_i])
		
		prop_now_ranked = rankdata(prop_now)
	
		th, weighted_omega_ranked = weightedOmegaTheta(ra_real, dec_real, prop_now_ranked, ra_rand, dec_rand, thmin, thmax, th_nbins)
		
		M_theta = np.array(mcfTheta(th, omega, weighted_omega_ranked)).reshape(len(th), 1)
				
		th_omega_mcfs = np.hstack((th_omega_mcfs, M_theta))
		
	return th_omega_mcfs
	
def _process_jackknife(args):

    jk_i, real_tab_arg, real_properties_arg, rand_tab_arg, thmin_arg, thmax_arg, th_nbins_arg, realracol_arg, realdeccol_arg, randracol_arg, randdeccol_arg, jackknife_samples_arg, working_dir = args

    try:
        if(jk_i == 0):
            real_tab_i, rand_tab_i = real_tab_arg, rand_tab_arg 
            result_file = os.path.join(working_dir, 'results', 'CFReal.txt')
            print("Working on the real sample: Nreal = %d, Nrand = %d" %(len(real_tab_i), len(rand_tab_i)))
        else:
            real_tab_i, rand_tab_i = jackknife_samples_arg[jk_i - 1]
            result_file = os.path.join(working_dir, 'results', 'jackknifes', 'CFJackknife_jk%d.txt' %jk_i)
            print("Working on the jackknife sample %d: Nreal = %d, Nrand = %d" %(jk_i, len(real_tab_i), len(rand_tab_i)))
		
        result_i = computeCF(real_tab_i, real_properties_arg, rand_tab_i, thmin_arg, thmax_arg, th_nbins_arg, realracol_arg, realdeccol_arg, randracol_arg, randdeccol_arg)

        np.savetxt(result_file, result_i, delimiter="\t",fmt='%f')
		
    except Exception as e:
        logging.error("Error processing jk_i = %d: %s", jk_i, e)
        return 1
		
    return 0
	
def runComputationAngular(real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, njacks_ra, njacks_dec, working_dir=os.getcwd(), realracol='RA',realdeccol='DEC',randracol='RA', randdeccol='Dec', omp=False):

    os.chdir(working_dir)
    os.makedirs(working_dir+os.path.sep+'biproducts',  exist_ok=True)
    os.makedirs(working_dir+os.path.sep+'results/jackknifes',  exist_ok=True)

    jackknife_samples = jkgen.makeJkSamples(real_tab, rand_tab, njacks_ra, njacks_dec, realracol, realdeccol, randracol, randdeccol, plot=False)

    n_jacks = njacks_ra * njacks_dec
	
    if(omp): 
        num_processes = cpu_count()
        print(f"Parallelizing with {num_processes} processes...")

        tasks = []
        for jk_i in range(n_jacks + 1):
            tasks.append((jk_i, real_tab, real_properties, rand_tab, thmin, thmax, th_nbins, realracol, realdeccol, randracol, randdeccol, jackknife_samples, working_dir))
            
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
	
	
	

