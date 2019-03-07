#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kavica is a HPC data science package which includes pre-processing and post-processing for cluster analysis.
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 17/12/2018

import os

from setuptools import setup, find_packages

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()

requirements = read_file('requirements.txt').split('\n')


setup(
    name="KAVICA",
    version="0.0a0",
    author="Kaveh Mahdavi",
    author_email="kavehmahdavi74@gmail.com",
    description="The KAVICA framework",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/kavehmahdavi/KAVICA",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering ',
        'Natural Language :: English',

    ],
    project_urls={
        "Documentation": "http://kavehmahdavi.github.io/kavica/",
        "Forum": "http://kavehmahdavi.github.io/kavica/",
        "Repository": "https://github.com/kavehmahdavi/kavica",
        "Issues": "https://github.com/kavehmahdavi/kavica/issues",
        "Author": "http://kavehmahdavi.github.io/kavica/",
    },
    zip_safe=False,
    keywords=['ICA', 'Feature Selection', 'Factor Analysis', 'ETL', 'Performance Analytics'],
    python_requires='>=3',
    package_data={},
    data_files=[],
    install_requires=requirements,
    extras_require={},
    entry_points={},
    ext_modules=[],
    cmdclass={},
    scripts=[],
)
