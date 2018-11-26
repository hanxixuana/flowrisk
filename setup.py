#!/usr/bin/env python
#
# Created by Xixuan on Nov 22, 2018
#

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='flowrisk',
    version='0.2.2',
    description='Order flow risk measures in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hanxixuana/flowrisk',
    author='Xixuan Han',
    author_email='xixuanhan@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'scipy'
    ],
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True
)
