try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'description': 'Socrates - Spectral Finite Element solver',
	'author': 'Alfredo Carella',
#	'url': 'URL to get it at.',
#	'download_url': 'Where to download it.',
	'author_email': 'alfredocarella@gmail.com',
	'version': '0.1',
	'install_requires': ['nose'],
	'packages': ['solverls'],
	'scripts': [],
	'name': 'Socrates'
}

setup(**config)
