try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Spectral / Finite Element library',
    'author': 'Alfredo Carella',
    'url': 'https://github.com/alfredocarella/speclib.git',
    # 'download_url': 'Where to download it.',
    'author_email': 'alfredocarella@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['solverls'],
    'scripts': [],
    'name': 'speclib'
}

setup(requires=['numpy', 'nose', 'matplotlib'], **config)
