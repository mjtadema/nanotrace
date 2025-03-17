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
"""
Implement built-in feature extractors.
Three main classes:
- Global features
    - can use as `*global_features`
- Frequency features
    - only `freq_by_power(t,y,*,n=8,fs)
    - :param n: number of frequencies to return
    - :param fs: signal sampling rate in Hz
- Sequence features
    - Use when the signal contains sequence information
    - combined in `*sequence_features`
"""

import logging

from functools import partial
from .decorators import catch_errors

import numpy as np
from scipy.signal import welch, periodogram
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


# Global features
@catch_errors()
def mean(t,y):
    return np.mean(y)

@catch_errors()
def std(t,y):
    return np.std(y)

@catch_errors()
def dt(t,y):
    return t[-1]-t[0]

@catch_errors()
def ldt(t,y):
    return np.log(dt(t,y))

@catch_errors()
def median(t,y):
    return np.median(y)

@catch_errors()
def _skew(t,y):
    x = np.linspace(0,1,100)
    k = gaussian_kde(y)
    return skew(k(x))

@catch_errors()
def kurt(t,y):
    x = np.linspace(0,1,100)
    k = gaussian_kde(y)
    return kurtosis(k(x))

@catch_errors(n=2)
def clst_means(t,y):
    fit_ = GaussianMixture(n_components=2).fit(y.reshape(-1, 1))
    return np.sort(fit_.means_.flatten())

global_features = [mean, std, ldt, median, _skew, kurt, clst_means]

# Frequency features
# TODO this should be done in a nice way _without_ the partial decorator
def freq_by_power(t,y,*,n=8,fs):
    """
    Calculate the PSD and return the n strongest frequencies
    :param n: number of frequencies to return
    :param fs: sampling rate
    """
    # when the sample is too small, welch's method becomes too flattened
    if len(y) > 256:
        fspectr = welch
    else:
        fspectr = periodogram
    X,Y = fspectr(y, fs=fs)
    return X[np.argsort(Y)][::-1][:n]

# Sequence features
def split(t,y,func,n):
    return [func(ts,ys) for ts,ys in zip(np.array_split(t,n), np.array_split(y,n))]

def _min(t,y):
    return np.min(y)

def _max(t,y):
    return np.max(y)

sequence_features = []
for f in (median, mean, std, _min, _max, _skew):
    pf = partial(split, func=f, n=8)
    pf.__name__ = f.__name__+'_split'
    sequence_features.append(pf)
