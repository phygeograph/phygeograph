import os
import re
import sys
import platform
from subprocess import CalledProcessError
import setuptools
from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
import io
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = '0.1.5'
kwargs = dict(
    name='phygeograph',
    version=__version__,
    url='https://github.com/phygeograph/phygeograph',
    author='Lianfa Li',
    author_email='phygeograph@gmail.com',
    description='Library for Physics-Aware Geo Graph Hybrid Network',
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    install_requires = [],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 6 - Mature',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    data_files=[],
)

# likely there are more exceptions, take a look at yarl example
try:
    setup(**kwargs)
except CalledProcessError:
    print('Failed to build extension!')
    del kwargs['ext_modules']
    setup(**kwargs)

