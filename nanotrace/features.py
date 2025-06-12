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
from __future__ import annotations
__copyright__ = """
Copyright 2025 Matthijs Tadema

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import functools as ft
from typing import Any

import numpy as np
from numpy import floating
from scipy.signal import welch, periodogram
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.mixture import GaussianMixture

from .decorators import catch_errors, partial

logger = logging.getLogger(__name__)


# Global features

@catch_errors()
def mean(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.mean(y)


@catch_errors()
def std(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.std(y)


@catch_errors()
def dt(t: np.ndarray, y: np.ndarray) -> float:
    return float(t[-1] - t[0])


@catch_errors()
def ldt(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.log(dt(t, y))


@catch_errors()
def median(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.median(y)


@catch_errors()
def _skew(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, 100)
    k = gaussian_kde(y)
    return skew(k(x))


@catch_errors()
def kurt(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, 100)
    k = gaussian_kde(y)
    return kurtosis(k(x))


@catch_errors(n=2)
def clst_means(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    fit_ = GaussianMixture(n_components=2).fit(y.reshape(-1, 1))
    return np.sort(fit_.means_.flatten())


global_features = [mean, std, ldt, median, _skew, kurt, clst_means]


# Frequency features

@partial
def psd_freq(t: np.ndarray, y: np.ndarray, *, n=8, fs: int) -> np.ndarray:
    """
    Calculate the PSD and return the n most prevalent frequencies
    :param n: number of frequencies to return
    :param fs: sampling rate
    """
    # when the sample is too small, welch's method becomes too flattened
    if len(y) > 256:
        fspectr = welch
    else:
        fspectr = periodogram
    X, Y = fspectr(y, fs=fs)
    return X[np.argsort(Y)][::-1][:n]


# Sequence features
def split(t, y, func, n) -> list[tuple[np.ndarray, np.ndarray]]:
    return [func(ts, ys) for ts, ys in zip(np.array_split(t, n), np.array_split(y, n))]


def _min(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.min(y)


def _max(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.max(y)


sequence_features = []
for f in (median, mean, std, _min, _max, _skew):
    pf = ft.partial(split, func=f, n=8)
    pf.__name__ = f.__name__ + '_split'
    sequence_features.append(pf)


def index_base(y):
    """
    Get the index of the base array y originates from
    """
    d = np.diff( # diff to get starts and ends of segments
        np.any( # get where each absolute difference is 0
            abs(
                # Subtract start and end value from the base
                y.base[:,None] - y[None,(0,-1)]
            ) == 0,
            axis=1),
        append=0)
    start = np.arange(len(d))[d==1][0]
    end = np.arange(len(d))[d==-1][-1]
    if end-start != len(y): raise Exception("could not find segment")
    return start,end


def gNDF(x, A, x0, sigma, B, C):
    """
    :param A: baseline
    :param x0: event location
    :param sigma: sigma of the distribution
    :param C: event block
    """
    E = -(np.abs(x - x0) / (2*sigma))**B
    return A*np.exp(E) + C


@catch_errors(n=3)
def peptide_fit(t,y):
    # expand the event to include some baseline for fitting
    s,e = index_base(y)
    l = e-s
    # Expand by twice the event length
    y = y.base[s-2*l:e+2*l].astype(np.float64)
    t = t.base[s-2*l:e+2*l].astype(np.float64)
    if len(y) > 5000:
        y,t = scipy.signal.resample(y, num=5000, t=t)
    # Estimate parameters
    x0 = np.mean(t)
    sigma = float(max(t)-min(t)) / 3
    beta = -2.72
    c = max(0,np.min(y))
    a = 1-c
    # Fit gNDF (doi 10.1021/acsomega.2c00871)
    popt, pcov = scipy.optimize.curve_fit(gNDF, t, y, p0=[a,x0,sigma,beta,c], maxfev=100, bounds=([0,t.min(),0,-np.inf,0],[1,t.max(),1,0,1]))
    a, x0, sigma, beta, c = popt
    # Calculate the dwelltime
    dt = 2 * sigma * scipy.special.gamma((1 / beta) + 1)
    # Return event characteristics mean block, log(dt) and sd
    yfit = gNDF(t, *popt)
    sd = np.std((y - yfit)[(x0-dt < t) & (t < x0+dt)])
    return c, np.log(dt), sd
