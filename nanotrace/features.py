"""
Implement built-in feature extractors.

Feature extractors are simple functions which must take
two 1d-arrays (t,y) as arguments, and return one or more values.

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
from scipy.optimize import curve_fit
from scipy.signal import welch, periodogram, resample
from scipy.special import gamma
from scipy.stats import gaussian_kde, skew as _skew, kurtosis as _kurtosis
from sklearn.mixture import GaussianMixture

from .decorators import catch_errors, partial

logger = logging.getLogger(__name__)


### Global features ###

@catch_errors()
def mean(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    """Calculate the mean of the y array"""
    return np.mean(y)


@catch_errors()
def std(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    """Calculate the standard deviation of the y array"""
    return np.std(y)


@catch_errors()
def dt(t: np.ndarray, y: np.ndarray) -> float:
    """Calculate the total time of a segment"""
    return float(t[-1] - t[0])


@catch_errors()
def ldt(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    """Calculate the log of the time of the segment"""
    return np.log(dt(t, y))


@catch_errors()
def median(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    """Calculate the median of y"""
    return np.median(y)


@catch_errors()
def skew(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the skewness of y but on a downsampled kde"""
    x = np.linspace(0, 1, 100)
    k = gaussian_kde(y)
    return _skew(k(x))


@catch_errors()
def kurt(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the kurtosis of y but on a downsampled kde"""
    x = np.linspace(0, 1, 100)
    k = gaussian_kde(y)
    return _kurtosis(k(x))


@catch_errors(n=2)
def clst_means(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Do a gaussian mixture fit with two components and return the means"""
    fit_ = GaussianMixture(n_components=2).fit(y.reshape(-1, 1))
    return np.sort(fit_.means_.flatten())


global_features = [mean, std, ldt, median, skew, kurt, clst_means]


### Frequency features ###

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
    X = np.asarray(X)
    return X[np.argsort(Y)][::-1][:n]


### Sequence features ###

def split(t, y, func, n) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate functions for split features"""
    return [func(ts, ys) for ts, ys in zip(np.array_split(t, n), np.array_split(y, n))]

@catch_errors()
def _min(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.min(y)

@catch_errors()
def _max(t: np.ndarray, y: np.ndarray) -> floating[Any]:
    return np.max(y)


sequence_features = []
for f in (median, mean, std, _min, _max, skew):
    pf = ft.partial(split, func=f, n=8)
    pf.__name__ = f.__name__ + '_split'
    sequence_features.append(pf)


def index_base(y: np.ndarray) -> tuple[int, int]:
    """
    Get the slice of the base array y originates from
    """
    #TODO move this to a common file once we have more stuff to put there
    d = np.diff( # diff to get starts and ends of segments
        np.any( # get where each absolute difference is 0
            abs(
                # Subtract start and end value from the base
                y.base[:,None] - y[None,(0,-1)]
            ) == 0,
            axis=1),
        append=0)
    start = int(np.arange(len(d))[d==1][0])
    end = int(np.arange(len(d))[d==-1][-1])
    if end-start != len(y): raise Exception("could not find segment")
    return start,end


def gNDF(x: np.ndarray, A: float, x0: float, sigma: float, B: float, C: float) -> np.ndarray:
    """
    Generalized NDF from doi.org/10.1021/acsomega.2c00871
    :param A: baseline
    :param x0: event location
    :param sigma: sigma of the distribution
    :param B: shape parameter
    :param C: event block
    :return: NDF over x
    """
    E = -(np.abs(x - x0) / (2*sigma))**B
    return A*np.exp(E) + C


@catch_errors(n=4)
def fit_gNDF(t: np.ndarray, y: np.ndarray):
    """
    Fit current data from y to the gNDF to characterize peptide blockage events
    :return: tuple(mean current, log(dwell time), event standard deviation, shape parameter beta)
    """
    if len(y) > 5000:
        # Resample if y is too large
        y, t = resample(y, num=5000, t=t)
    # Fit gNDF (doi 10.1021/acsomega.2c00871)
    x = np.linspace(0, 1, len(y))  # need to normalize x for a good fit
    popt, pcov, *_ = curve_fit(
        gNDF, x, y,
        maxfev=100,  # Low limit of function evaluations, if the fit is not fast assume it's a bad fit
        bounds=(
            [0, 0, 0, -np.inf, 0],  # low
            [1, 1, 1, -1, 1]),  # high
        p0=[0.5, 0.5, 1 / 3, -2.72, 0.5]  # initial values
    )
    # Convert back to real time
    timespan = (t[-1] - t[0])

    a, x0, sigma, beta, c = popt
    # Calculate the dwelltime using the gamma function
    dt = 2 * sigma * gamma((1 / beta) + 1) * timespan

    x0 = t[0] + x0 * timespan

    # Return event characteristics mean block, log(dt) and sd
    yfit = gNDF(x, *popt)
    sd = np.std((y - yfit)[(x0 - dt < t) & (t < x0 + dt)])
    return c, np.log(dt), sd, np.log(-beta)
