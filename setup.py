#!/usr/bin/env python
#
# Created by Xixuan on Nov 22, 2018
#

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='FlowRisk',
    version='0.1',
    description='Order flow risk measures in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hanxixuana/flowrisk',
    author='Xixuan Han',
    author_email='xixuanhan@gmail.com',
    license='MIT',
    packages=['flowrisk'],
    install_requires=[
      'numpy', 'pandas', 'matplotlib', 'scipy'
    ],
    zip_safe=True
)
