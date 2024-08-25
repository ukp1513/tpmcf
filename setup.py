from setuptools import setup

setup(name='2mpcf',
      description='A package for computing and analyzing two-point and marked correlation functions.',
	version='0.1',
	author='Unnikrishnan Sureshkumar',
	author_email='unnikrishnan.sureshkumar@wits.ac.za',
	packages=['2mpcf'],
	include_package_data=True,
	package_data={'2mpcf': ['resources/*.ttf']},
	install_requires=['numpy',
      			'matplotlib',
        		'scipy',
        		'astropy',
        		'treecorr',
        		'healpy',
    			],		
	url='https://github.com/ukp1513/2mpcf',
)
