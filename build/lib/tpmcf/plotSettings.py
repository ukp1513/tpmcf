"""

This module contains global plot settings. 

For any routine that makes a plot using matplotlib, call plotSettings.update_rcParams() first.

"""

import pylab as plt
import matplotlib as mpl
from cycler import cycler
import os
from matplotlib import font_manager as fm

def update_rcParams(dict={}):
	default = {}
	for tick in ('xtick', 'ytick'):
		default['{0}.major.size'.format(tick)] = 6
		default['{0}.minor.size'.format(tick)] = 2
		default['{0}.major.width'.format(tick)] = 1
		default['{0}.minor.width'.format(tick)] = 1
		default['{0}.labelsize'.format(tick)] = 12
		default['{0}.direction'.format(tick)] = 'in'
	default['xtick.top'] = True
	default['ytick.right'] = True
	default['axes.linewidth'] = 1
	default['axes.titlesize'] = 12
	default['axes.labelsize'] = 16
	default['font.size'] = 22
	#default['font.family'] = 'sans-serif'
	default['legend.fontsize'] = 12
	default['lines.linewidth'] = 2
	default['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
	default['mathtext.fontset'] = 'cm'
	
	default['font.family'] = 'CMU Sans Serif'
		
	for key in default:
		plt.rcParams[key] = default[key]
	# Overwrite with any user-specified parameters
	for key in dict:
		plt.rcParams[key] = dict[key]
	
	print("Plot settings imported")
	
def set_tick_font_to_serif(ax):
    """Apply serif font only to tick labels of a given axis."""
    for label in ax.get_xticklabels():
        label.set_fontname("serif")  # Set serif font for tick labels
    for label in ax.get_yticklabels():
        label.set_fontname("serif")  # Set serif font for tick labels

update_rcParams()
