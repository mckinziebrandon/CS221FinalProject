#!/usr/bin/env python3.6

import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='final_project',
    description='Final Project',
    long_description=readme,
    author='Brandon McKinzie',
    author_email='bmckinz@stanford.edu',
    packages=find_packages(),
    install_requires=[
        'colorama',
        'dataclasses==0.6',
        'inflection==0.3.1',
        'Keras==2.2.4',
        'Keras-Applications==1.0.6',
        'Keras-Preprocessing==1.0.5',
        'lxml==4.2.5',
        'nltk==3.4',
        'numpy==1.16.2',
        'Pympler==0.6',
        'PyYAML==3.12',
        'tensorflow==2.5.0rc0'
    ])


