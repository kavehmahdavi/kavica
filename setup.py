#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAVICA: Powerful Python Cluster Analysis and Inference Toolkit

Note:
    PyPI:
        python setup.py sdist
        twine upload --repository testpypi dist/*
        twine upload dist/*


Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 18/10/2022
"""
import os
from setuptools import setup, find_packages


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()


requirements = read_file('requirements.txt').split('\n')

setup(
    name="KAVICA",
    version="1.3.4",
    author="Kaveh Mahdavi",
    author_email="kavehmahdavi74@yahoo.com",
    description="KAVICA: Powerful Python Cluster Analysis and Inference Toolkit",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/kavehmahdavi/kavica",
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
        "Forum": "http://kavehmahdavi.github.io/",
        "Repository": "https://github.com/kavehmahdavi/kavica",
        "Issues": "https://github.com/kavehmahdavi/kavica/issues",
        "Author": "http://kavehmahdavi.github.io/",
    },
    zip_safe=False,
    keywords=['Cluster Inference System', 'Feature Selection', 'Factor Analysis', 'Parser',
              'Clustering', 'Unsupervised', 'Self-organizing map', 'Organization Component Analysis',
              'Feature Space Curvature Map', 'Multiline Transformation'],
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
