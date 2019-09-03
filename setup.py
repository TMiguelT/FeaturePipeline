#!/usr/bin/env python
from setuptools import setup, find_packages

# Always prefer setuptools over distutils
setup(
    name='FeaturePipeline',
    version='0.0.1',
    description='Framework from extracting feature vectors from text for machine learning',
    url='https://github.com/TMiguelT/FeaturePipeline',
    author='Michael Milton',
    author_email='michael.r.milton@gmail.com',
    license='GPLv3',
    test_suite='test',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'nltk',
        'regex'
    ],
    extras_require={
        'dev': [
            'pytest'
        ]
    }
)
