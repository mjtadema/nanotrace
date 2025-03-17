"""
author: Matthijs Tadema

Pipeline:
---------
`pipeline` is a python library for automated nanopore electrophysiology (1d timeseries)
manipulation and feature extraction. The central class is the `Pipeline` class, imported from the main module:
`from pipeline import Pipeline`.
The main module also imports ABF from pyabf for convenience.
"""
from .pipeline import Pipeline
from pyabf import ABF