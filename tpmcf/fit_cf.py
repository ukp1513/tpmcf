import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os
import shutil
from scipy.optimize import curve_fit
import bokeh.palettes as bp
from . import integral_constrain

from matplotlib import font_manager as fm, rcParams
import socket

if(os.environ['THIS_PLATFORM'] == 'hp455'):
	fpath = "/home/krishna/Dropbox/fonts/cmunss.ttf"
	source_dir = '/home/krishna/krishna_work/DES_MCF'
elif(os.environ['THIS_PLATFORM'] == 'hippo'):
	fpath = "/home/ukp1513/fonts/cmunss.ttf"
	source_dir = "/data/ukp1513/des_mcf/"
elif(os.environ['THIS_PLATFORM'] == 'plgrid'):
	fpath = '/net/people/plgrid/plgukp1513/fonts/cmunss.ttf'
	source_dir = '/net/ascratch/people/plgukp1513/des_mcf/data'
elif(os.environ['THIS_PLATFORM'] == 'chpc'):
	fpath = '/home/usureshkumar/fonts/cmunss.ttf'
	source_dir = '/home/usureshkumar/lustre/des'
else:
        print("Platform not found! Exiting...")
        exit(0)

	
prop = fm.FontProperties(fname=fpath,size=12,math_fontfamily='stixsans')
prop_big = fm.FontProperties(fname=fpath,size=14,math_fontfamily='stixsans')
prop_small = fm.FontProperties(fname=fpath,size=12,math_fontfamily='stixsans')
prop_tiny = fm.FontProperties(fname=fpath,size=7,math_fontfamily='stixsans')
fname = os.path.split(fpath)[1]

plt.style.use('classic')

def covmat_to_corrmat(covariance):
	v = np.sqrt(np.diag(covariance))
	outer_v = np.outer(v, v)
	correlation = covariance / outer_v
	correlation[covariance == 0] = 0
	return correlation
	
def angularCF_model(theta, A, gam):
	return A*pow(theta, 1-gam)
	
def redshift3dCF_model(s, s0, gam):
	return pow((s/s0), (-1*gam))

def fitCFMcf(stattype, sepmin, sepmax, sepmin_tofit, sepmax_tofit, real_tab, rand_tab, real_properties, to_svd_filter=0, to_hartlap_corr=0, fit_2pcf = 1, work_on_mcf = 1, dir_name=os.getcwd(), plotxscale='log', plotyscale='log', ignore_negatives = True, realracol='RA', realdeccol='DEC', randracol='RA', randdeccol='Dec', compute_IC = True):

	if(stattype == 'angular'):

		cfxlabel = r"$\theta \,\,\, (deg)$"
		cfylabel = r"$\omega(\theta)$"
		mcfxlabel = r"$\theta \,\,\, (deg)$"
		mcfylabel = r"$M (\theta)$"
		cffig_name = "fig_angularCF.png"
		mcffig_name = "fig_angularMCF.png"

	elif(stattype == '3d_redshift'):

		cfxlabel = r"$s \,\,\, (Mpc/h)$"
		cfylabel = r"$\xi(s)$"
		mcfxlabel = r"$s \,\,\, (Mpc/h)$"
		mcfylabel = r"$M (s)$"
		cffig_name = "fig_3DRedshiftCF.png"
		mcffig_name = "fig_3DRedshiftMCF.png"
		
	os.chdir(dir_name)

	if(to_svd_filter == 0):
		print("\nSVD correction NOT done!")
	else:
		print("\nSVD correction done!")

	if(to_hartlap_corr == 0):
		print("\nHartlap correction NOT done!")
	else:
		print("\nHartlap correction done!")

	
	marks = real_properties
	n_marks = len(marks)

	# REMOVING PREVIOUS INV COVAR FILES IF EXIST
	

	if os.path.exists("biproducts/inv_cov_mat.txt.txt"):
		os.remove("biproducts/inv_cov_mat.txt.txt")
	if os.path.exists("biproducts/inv_corr_mat_SVD.txt"):
		os.remove("biproducts/inv_corr_mat_SVD.txt")
		
	# CHECKING BOOTSTRAP/JACKKNIFE

	cwd_fullpath = os.getcwd()
	cwd_split=cwd_fullpath.split('/')
	sample_name=cwd_split[-1].upper()

	print("\n--------------------------------------------")
	print("\nFitting sample %s....\n" %(sample_name))
	print("--------------------------------------------\n")

	is_bs=0
	is_jk=0

	if(os.path.isdir("results/bootstraps")):
		is_bs=1	
	if(os.path.isdir("results/jackknifes")):
		is_jk=1
		
	# CREATING CFRealAll.txt for JK/BS copies

	sep=np.loadtxt('results/CFReal.txt')[:,0]
	CFReal=np.loadtxt('results/CFReal.txt')[:,1]

	total_nbins = len(sep)
	ncopies=len([f for f in os.listdir('results/jackknifes') if not f.startswith('.')])

	print('Nr. of total bins: ', total_nbins)
	print('Nr. of JK/BS copies: ', ncopies)

	CFRealAll_tofile = np.ndarray(shape=(ncopies+2,total_nbins), dtype=float)

	CFRealAll_tofile[:][0]=sep
	CFRealAll_tofile[:][1]=CFReal

	for copy in range(ncopies):
		CFJK=np.loadtxt('results/jackknifes/CFJackknife_jk%d.txt' %(copy+1))[:,1]
		CFRealAll_tofile[:][copy+2]=CFJK

	np.savetxt("results/CFRealAll.txt",np.transpose(CFRealAll_tofile),delimiter="\t",fmt='%f')	


	if(work_on_mcf == 1):
		# COLLECTING MCFs

		print("Marks : ", marks)


		for mark_i, mark in enumerate(marks):
			mcfReal = np.loadtxt('results/CFReal.txt')[:,mark_i+2]
			mcfRealAll_tofile = np.ndarray(shape=(ncopies+2,total_nbins), dtype=float)
			mcfRealAll_tofile[:][0]=sep
			mcfRealAll_tofile[:][1]=mcfReal
			for copy in range(ncopies):
				mcfJK = np.loadtxt('results/jackknifes/CFJackknife_jk%d.txt' %(copy+1))[:,mark_i+2]
				mcfRealAll_tofile[:][copy+2]=mcfJK


			np.savetxt("results/mcfRealAll_%s.txt" %mark,np.transpose(mcfRealAll_tofile),delimiter="\t",fmt='%f')		

	# FILTERING NAN AND INF VALUES

	CFRealAll = np.loadtxt('results/CFRealAll.txt')

	nrows,ncols = CFRealAll.shape

	total_nbins = nrows
	ncopies = ncols-2

	nrows=total_nbins
	ncols=ncopies

	filter_index_CF = []
	for i in range(0,nrows):
		if(ignore_negatives == True):
			if(np.isnan(CFRealAll[i,1]).any() == True or np.isinf(CFRealAll[i,1]).any() == True or CFRealAll[i,1]<0.):
				filter_index_CF.append(i)
			else: 
				for j in range(ncols):
					if(np.isnan(CFRealAll[i,j]).any() == True or np.isinf(CFRealAll[i,j]).any() == True):
						filter_index_CF.append(i)
		else:
			if(np.isnan(CFRealAll[i,1]).any() == True or np.isinf(CFRealAll[i,1]).any() == True):
				filter_index_CF.append(i)
			else: 
				for j in range(ncols):
					if(np.isnan(CFRealAll[i,j]).any() == True or np.isinf(CFRealAll[i,j]).any() == True):
						filter_index_CF.append(i)
		if(CFRealAll[i,0] < sepmin or CFRealAll[i,0] > sepmax):
			filter_index_CF.append(i)

	filter_index_CF=list(set(filter_index_CF))	

	#REMOVING NAN BINS FROM CF FILE
	CFRealAll=np.delete(CFRealAll, filter_index_CF, axis=0)

	nbins_CF=total_nbins-len(filter_index_CF)

	print("Number of CF bins with non-nan values:", nbins_CF)
	np.savetxt('results/CFRealAll_filtered.txt',CFRealAll,delimiter='\t',fmt='%f')

	#REMOVING NAN BINS FROM mcf FILE
	if(work_on_mcf == 1):
		for mark_i, mark in enumerate(marks):
			mcfRealAll = np.loadtxt('results/mcfRealAll_%s.txt' %mark)
			mcfRealAll=np.delete(mcfRealAll, filter_index_CF, axis=0)
			np.savetxt('results/mcfRealAll_%s_filtered.txt' %mark,mcfRealAll,delimiter='\t',fmt='%f')


	#FILTERING TO FIT BINS

	filter_index_CF_tofit = []
	for i in range(0,nbins_CF):
		if(CFRealAll[i,0] < sepmin_tofit or CFRealAll[i,0] > sepmax_tofit):
			filter_index_CF_tofit.append(i)
	filter_index_CF_tofit=list(set(filter_index_CF_tofit))
	nbins_CF_tofit = nbins_CF-len(filter_index_CF_tofit)

	print("Number of bins used for CF fitting ", nbins_CF_tofit)

	CFRealAll_tofit=np.delete(CFRealAll, filter_index_CF_tofit, axis=0)


	np.savetxt('results/CFRealAll_filtered_tofit.txt', CFRealAll_tofit,delimiter='\t',fmt='%f')



	# COMPUTING COVARIANCE MATRIX AND PLOTTING ALL BINS

	fig,ax_now=plt.subplots(nrows=1,ncols=1,sharex=False)
	fig.set_size_inches(5,5)

	sep_toPlot = np.loadtxt('results/CFRealAll_filtered.txt')[:,0]
	CF_toPlot = np.loadtxt('results/CFRealAll_filtered.txt')[:,1]


	if(ncopies > 0):

		allCopiesCFs = CFRealAll[:,2:ncopies+2]

		if(is_jk==1):
			cov_mat=np.cov(allCopiesCFs, bias=True)	# C = (Njk-1)/Njk x SUM
			cov_mat = (ncopies-1)*cov_mat

		if(is_bs==1):
			cov_mat=np.cov(allCopiesCFs, bias=False)	# C = (Njk-1)/Njk x SUM
			
		CF_err_toPlot = np.sqrt(np.diag(cov_mat))
		
	else:
		CF_err_toPlot = [0 for i in range(len(sep_toPlot))]


	plt.errorbar(sep_toPlot, CF_toPlot, CF_err_toPlot,ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='white',ecolor='black',elinewidth=1)

	#final_path = 'finals'
	final_path = 'finals_%s_%s' %(str(sepmin_tofit).replace(".","p"),str(sepmax_tofit).replace(".","p"))
	if not os.path.exists(final_path):
	    os.makedirs(final_path)
	else:
	    shutil.rmtree(final_path)           # Removes all the subdirectories!
	    os.makedirs(final_path)

	np.savetxt(final_path+os.path.sep+'final_CF_toPlot.txt', np.transpose([sep_toPlot, CF_toPlot, CF_err_toPlot]), fmt='%f', delimiter='\t')


	if(fit_2pcf == 1):

		if(ncopies > 0):

			allCopiesCFs_tofit = CFRealAll_tofit[:,2:ncopies+2]

			if(is_jk==1):
				cov_mat=np.cov(allCopiesCFs_tofit, bias=True)	# C = (Njk-1)/Njk x SUM
				cov_mat = (ncopies-1)*cov_mat

			if(is_bs==1):
				cov_mat=np.cov(allCopiesCFs_tofit, bias=False)	# C = (Njk-1)/Njk x SUM

			if(to_svd_filter==1):

				corr_mat = covmat_to_corrmat(cov_mat)

				# FITTING USING SVD 

				U, Dvector, UT = np.linalg.svd(corr_mat)	# C = U D UT

				Dinv_vec = []

				neff=nbins_CF_tofit

				for i in Dvector:
					if(to_svd_filter==1):
						if(i < np.sqrt(2./ncopies)):
							neff-=1
							Dinv_vec.append(0.0)
						else:
							Dinv_vec.append(1./i)	
					else:
							Dinv_vec.append(1./i)

				Dinv = np.diag(Dinv_vec)
				inv_corr_mat_SVD = np.matmul(U,np.matmul(Dinv,UT))

				if(to_hartlap_corr == 1):
					hartlap_factor = (ncopies-nbins_CF_tofit-2)/(ncopies-1)
					print("Hartlap factor ",hartlap_factor,"\n")
					inv_corr_mat_SVD = hartlap_factor*inv_corr_mat_SVD
			    

			else:
				neff=nbins_CF_tofit
				try:
					inv_cov_mat = np.linalg.inv(cov_mat)
				except np.linalg.LinAlgError:
					print("\nIssue taking inverse of covariance matrix! Returning without fitting...")
					return

				if(to_hartlap_corr == 1):
					hartlap_factor = (ncopies-nbins_CF_tofit-2)/(ncopies-1)
					print("Hartlap factor ",hartlap_factor,"\n")
					inv_cov_mat = hartlap_factor*inv_cov_mat

			fEff=open("biproducts/effective_bins.txt","w")
			fEff.write(str(neff))
			fEff.close()

			if(to_svd_filter==1):
				np.savetxt("biproducts/inv_corr_mat_SVD.txt",np.transpose(inv_corr_mat_SVD),delimiter="\t",fmt='%f')
			else:
				np.savetxt("biproducts/inv_cov_mat.txt",np.transpose(inv_cov_mat),delimiter="\t",fmt='%f')
				np.savetxt("biproducts/cov_mat.txt",np.transpose(cov_mat),delimiter="\t",fmt='%f')


		sep_toFit = np.loadtxt('results/CFRealAll_filtered_tofit.txt')[:,0]
		CF_toFit = np.loadtxt('results/CFRealAll_filtered_tofit.txt')[:,1]
		inv_cov_mat_toFit = inv_cov_mat
		try:
			cov_mat_toFit = np.linalg.inv(inv_cov_mat_toFit)
		except np.linalg.LinAlgError:
			print("\nIssue taking inverse of covariance matrix! Returning without fitting...")
			return
		CF_err_toFit = np.sqrt(np.diag(cov_mat_toFit))

		np.savetxt(final_path+os.path.sep+'final_CF.txt', np.transpose([sep_toFit, CF_toFit, CF_err_toFit]), fmt='%f', delimiter='\t')
		
		# FIT USING CURVE_FIT

		if(stattype == 'angular'):
			
			try:
				popt, pcov = curve_fit(angularCF_model, sep_toFit, CF_toFit, sigma=cov_mat_toFit)
			except RuntimeError as e:
				print("\nProblem fitting curve: RuntimeError")
				print(f"Error message: {e}")
				return
			except Exception as e:
				print("\nProblem fitting curve: General Exception")
				print(f"Error message: {e}")
				return
			A_curve, A_err_curve, gam_curve, gam_err_curve = popt[0],np.sqrt(pcov[0,0]),popt[1],np.sqrt(pcov[1,1])
			print('Curve fitting parameters:\nA = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %(A_curve, A_err_curve, gam_curve, gam_err_curve))
			best_fit_model_curve=angularCF_model(sep_toFit, A_curve, gam_curve)
			plt.errorbar(sep_toFit, CF_toFit, CF_err_toFit,ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)
			label = r"$\omega(\theta)=A \theta^{1-\gamma}$" + "\n" + r"$A = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
			plt.plot(sep_toPlot, angularCF_model(sep_toPlot, A_curve, gam_curve), color='red',label=label %(A_curve, A_err_curve, gam_curve, gam_err_curve))
			
			# WRITING TO FILES

			np.savetxt(final_path+os.path.sep+'CF_fit_params.txt', [A_curve, A_err_curve, gam_curve, gam_err_curve], fmt='%f', delimiter='\n')
			np.savetxt(final_path+os.path.sep+'sepFitRange.txt', [sepmin_tofit, sepmax_tofit], fmt='%f', delimiter='\n')
			
			
			if(compute_IC == True):
				IC = integral_constrain.computeICAngular(randgalaxies=rand_tab, A=A_curve, gamma=gam_curve, randracol=randracol, randdeccol=randdeccol)
				print("Integral Constrain = %f" %IC)
				with open(final_path+os.path.sep+'IC.txt', 'w') as file:
					file.write(str(IC))
				
				
		elif(stattype == '3d_redshift'):
		
			try:
				popt, pcov = curve_fit(redshift3dCF_model, sep_toFit, CF_toFit, sigma=cov_mat_toFit)
			except RuntimeError as e:
				print("\nProblem fitting curve: RuntimeError")
				print(f"Error message: {e}")
				return
			except Exception as e:
				print("\nProblem fitting curve: General Exception")
				print(f"Error message: {e}")
				return
			s0_curve, s0_err_curve, gam_curve, gam_err_curve = popt[0],np.sqrt(pcov[0,0]),popt[1],np.sqrt(pcov[1,1])
			print('Curve fitting parameters:\ns0 = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %(s0_curve, s0_err_curve, gam_curve, gam_err_curve))
			best_fit_model_curve=redshift3dCF_model(sep_toFit, s0_curve, gam_curve)
			plt.errorbar(sep_toFit, CF_toFit, CF_err_toFit,ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)
			label = r"$\xi(s)=(s/s_0) \theta^{-\gamma}$" + "\n" + r"$s_0 = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
			plt.plot(sep_toPlot, redshift3dCF_model(sep_toPlot, s0_curve, gam_curve), color='red',label=label %(s0_curve, s0_err_curve, gam_curve, gam_err_curve))

		
			# WRITING TO FILES

			np.savetxt(final_path+os.path.sep+'CF_fit_params.txt', [s0_curve, s0_err_curve, gam_curve, gam_err_curve], fmt='%f', delimiter='\n')
			np.savetxt(final_path+os.path.sep+'sepFitRange.txt', [sepmin_tofit, sepmax_tofit], fmt='%f', delimiter='\n')
			
			
			if(compute_IC == True):
				print("IC Computation is not coded for 3d...") #TODO


	plt.xscale(plotxscale)
	plt.yscale(plotyscale)
	plt.xlabel(cfxlabel,labelpad=10, fontproperties=prop_big)
	plt.ylabel(cfylabel,labelpad=0.5, fontproperties=prop_big)
	plt.legend(prop=prop)
	plt.savefig(cffig_name , dpi=300, bbox_inches = 'tight')
	plt.close()

		
	# PLOTTING MCF

	if(work_on_mcf == 1):


		if(len(marks) >= 10):
			colors = bp.d3['Category10'][10]+bp.d3['Category10'][10]
		elif(len(marks) >= 3):
			colors=bp.d3['Category10'][len(marks)]
		elif(len(marks) ==2):
			colors=['#ff7f0e','#2ca02c']
		else:
			colors=['black']
			
		markers=['s','H','v','+','x','d','s','^','p','D','o','h','*','H','v','+','x','d']

		fig,ax_now=plt.subplots(nrows=1,ncols=1,sharex=False)
		fig.set_size_inches(5,5)

		for mark_i, mark in enumerate(marks):
			mcfRealAll_toplot = np.loadtxt('results/mcfRealAll_%s_filtered.txt' %mark)
			sep_mcf = mcfRealAll_toplot[:,0]
			mcf = mcfRealAll_toplot[:,1]
			
			allCopiesMcfs = mcfRealAll_toplot[:,2:ncopies+2]
			
			mcf_err = np.std(allCopiesMcfs, axis=1)
			
			np.savetxt(final_path+os.path.sep+'final_mcf_%s.txt' %mark, np.transpose([sep_mcf, mcf, mcf_err]), fmt='%f', delimiter='\t')
			
			ax_now.errorbar(sep_mcf,mcf,mcf_err,color=colors[mark_i],capsize=3,ms=6,marker=markers[mark_i],mew=1.0,mec=colors[mark_i],mfc=colors[mark_i],ecolor=colors[mark_i],elinewidth=1,lw=1.0,label="%s" %(marks[mark_i]))
			
			ax_now.axhline(y=1, color='black', linestyle='dashed')

		plt.xscale(plotxscale)
		plt.xlabel(mcfxlabel,labelpad=10, fontproperties=prop_big)
		plt.ylabel(mcfylabel,labelpad=0.5, fontproperties=prop_big)
		plt.legend(numpoints=1,frameon=False,loc=0,prop=prop_tiny)
			
		plt.grid(False)
		plt.subplots_adjust(hspace=0.0,wspace=0.2)
		plt.savefig(mcffig_name, dpi=300, bbox_inches = 'tight')
		plt.close()
		
	return None
		
	

