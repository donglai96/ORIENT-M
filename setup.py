# @Author : Donglai
# @Time   : 11/23/2020 11:33 PM
# @Email  : dma96@atmos.ucla.edu donglaima96@gmail.com

"""A setuptools based setup module."""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open

setup(
    name='ORIENTM',
    version='1.0',
    description='Electron flux model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/donglai96/ORIENTM',
    packages=find_packages(),
    author='Donglai Ma',
    author_email='dma96@atmos.ucla.edu',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.8',
                 ],
    keywords='radiation belt modeling',

    install_requires=['numpy','tensorflow >=2.0','pandas','matplotlib','scipy','mechanize','pyspedas'],
    python_requires='>=3.5',
    include_package_data=True,
)

