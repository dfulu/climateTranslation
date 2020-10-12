#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

setup_requirements = [ ]

setup(
    author="James Fulton",
    author_email='djamesfulton@yahoo.co.uk',
    python_requires='>=3.6',
    description="Experiments on translating between different models of the climate",
    install_requires=requirements,
    license="CC BY-SA 4.0",
    long_description=readme,
    include_package_data=True,
    keywords='climatetranslation',
    name='climatetranslation',
    packages=find_packages(include=['climatetranslation', 'climatetranslation.*']),
    setup_requires=setup_requirements,
    url='https://github.com/dfulu/climateTranslation',
    version='0.1.0',
    zip_safe=False,
)
