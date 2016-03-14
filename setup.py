"""
Helper library for implementing models using the Hume interface

"""
import os
from setuptools import find_packages, setup

#   Allow setup.py to be run from any path
ORIG_DIR = os.getcwd()
try:
    MODULE_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    #   Missing '__file__': Must manually change directories before running this
    MODULE_DIR = ORIG_DIR
os.chdir(MODULE_DIR)

with open('requirements.txt', 'rbU') as f_in:
    REQS = ( l.strip() for l in f_in )
    REQS = ( l for l in REQS if l )
    REQS = list(REQS)

setup(
    name='model_tools',
    version='0.1.0',
    license='BSD',
    author='MJ Berends',
    author_email='mj@dose.com',
    description='Helpers for implementing models using the Hume interface',
    long_description=__doc__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    entry_points={},
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)

#   Change back to the original directory.
if ORIG_DIR != MODULE_DIR:
    os.chdir(ORIG_DIR)
