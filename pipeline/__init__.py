"""
author: Matthijs Tadema

Pipeline:
---------
`pipeline` is a python library for automated nanopore electrophysiology (1d timeseries)
manipulation and feature extraction. The central class is the `Pipeline` class, imported from the main module:
`from pipeline import Pipeline`.
The main module also imports ABF from pyabf for convenience.
"""
# Copyright 2025 Matthijs Tadema
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.WARNING)
from .pipeline import Pipeline
from pyabf import ABF