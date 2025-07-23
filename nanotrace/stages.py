from __future__ import annotations

import warnings
from typing import Generator, Sequence, Literal

from .learn import Predictor

"""
Pipeline stages
---------------
Functions to be used as stages in a pipeline
These must take time and current arrays as positional arguments
and return an iterator of new time and current arrays

Currently implemented stages:
-----------------------------
 `lowpass(cutoff_fq, fs, order=10)`:
    Apply a lowpass filter with `cutoff_fq` as the cutoff frequency in Hz, `fs`
    as the sampling rate and `order` as the order of the filter. The sampling rate
    can be extracted from an abf file using `ABF().SampleRate`

 `as_ires(minsamples=1000)`        :
    Calculate the _residual current_ (Ires) from the baseline.
    Automatically detects the baseline based on a binning approach.
    `minsamples` determines how many samples a bin needs to be considered
    a proper level and not just a fast current "spike".

 `trim(left=0, right=1)`           :
    Trim off this many samples from the `left` or the `right` side.
    If the sampling rate was assigned to a variable named `fs`,
    you can use this to calculate how many _seconds_ to trim off each side using `nseconds * fs`.

 `switch()`                        :
    Segment a gapfree nanotrace based on large, short, current spikes cause by manual voltage switching.

 `threshold(lo,hi)`                :
    Segment an input segment by consecutive stretches of current between `lo` and `hi`.

 `levels(n, tol=0, sortby='mean')` :
    Detect sublevels by fitting a [gaussian mixture model]
    (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
    Use `n` to set the number of gaussians to fit, `tol` is a number between 0 and 1
    and controls how much short spikes are tolerated. `sortby` controls how the gaussians
    are labeled, can be sorted by "mean" or by "weight" (weight being the height of the gaussian).

 `volt(c, v)` :
    Select a part of a nanotrace where the voltage `v` matches the control voltage array `c`.

Custom stages:
--------------
    Each stage is a callable that takes _only_ a time array and a current array as positional arguments.
    Additional parameters to stages can be passed by using `functools.partial` or by decorating
    a function with the `partial` decorator included in this library.
    These are typically defined as generators, yielding zero, one or more "segments" derived from the input.
    If the stage does not yield zero segments, it acts as a filter(1).
    A stage can yield a single segment, this is the case with the lowpass filter for example(2).
    Most stages yield several segments and thus the tree is constructed step by step(3).

    Example 1:
    --------::

        def stage(time, current):
            if condition:
                yield time, current

    Example 1:
    --------::

        def stage(time, current):
            new_current = f(current)
            yield time, current

    Example 3:
    --------::

        def stage(time, current):
            for new_time, new_current in f(time, current):
                yield new_time, new_current

"""
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

import re
from functools import wraps

import numpy as np
from pyabf import ABF
from scipy import signal
from scipy.signal import find_peaks, fftconvolve
from sklearn.mixture import GaussianMixture
from numba import njit

from .exception import StageError, BadBaseline
from .decorators import partial


# Utilities
def outliers(data: np.ndarray, m: float=2.) -> bool:
    """
    Reject outliers based on median absolute deviation (MAD).
    Inspired by stack overflow (https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list)

    :param data: input data
    :param m: modified z-score
    :return: boolean array where outliers are False
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return s>m


def baseline(y: np.ndarray, min_samples: int=1000,
             lo: float=50., hi: float=150.) -> tuple[np.floating, np.floating] | None:
    """
    Calculate baseline between a specified range.
    Outliers are rejected using MAD criteria.
    median and standard deviation of baseline are returned

    :param y: data
    :param min_samples: Minimum number of samples to consider
    :param lo: Minimum amplitude to consider
    :param hi: Maximum amplitude to consider
    :return: Baseline and standard deviation
    """
    thres = (lo < y) & (y < hi)
    inliers = ~outliers(y[thres], m=2)
    clean = y[thres][inliers]
    if len(clean) >= min_samples:
        return np.median(clean), np.std(clean)
    else:
        raise BadBaseline(f"#samples {len(clean)} < min samples {min_samples}")


def baseline_from_sweeps(abf: ABF, nbins: int=10, maxfail: int=0.5, **kwargs) -> tuple[float, float] | None:
    """
    Try to calculate the baseline from each sweep,
    then bin them and take the mean of the highest bin.
    This avoids baseline miscalculation when there is little to no
    baseline present.

    :param abf: ABF object
    :param nbins: Number of bins to use
    :param maxfail: Maximum number of failures (fraction)
    :return: Baseline and standard deviation
    """
    baselines = []
    fails = 0
    for i in range(abf.sweepCount):
        abf.setSweep(i)
        try:
            bl, sd = baseline(abf.sweepY, **kwargs)
            baselines.append((bl, sd))
        except (TypeError, IndexError, BadBaseline) as e:
            fails += 1
            if fails > abf.sweepCount * maxfail:
                raise Exception("too many baseline failures") from e

    baselines = np.asarray(baselines)

    counts, edges = np.histogram(baselines, bins=nbins)
    bins = np.array([a + b / 2 for a, b in zip(edges[:-1], edges[1:])])

    digi = np.digitize(baselines[:, 0], bins=bins)
    highest = baselines[digi == max(digi)]
    return np.mean(highest, axis=0).astype(float)


def smooth_pred(y: np.ndarray, fit: Predictor, tol: float) -> np.ndarray:
    """
    Smoothen a gaussian mixture prediction

    :param y: input data
    :param fit: predictor to smoothen
    :param tol: tolerance parameter determines sliding window size
    :return: smoothed data
    """
    # tol between 0 and 1?

    if tol <= 0:
        # Special case, do the regual prediction
        return fit.predict(y.reshape(-1, 1))
    elif tol > 1:
        tol = 1
    proba = fit.predict_proba(y.reshape(-1, 1))
    klen = int((len(y) / 10) * tol)
    if not klen > 0:
        raise StageError("Segment too small for smoothing")
    kernel = np.full((klen, proba.shape[1]), 1 / klen)
    pred = np.argmax(fftconvolve(proba, kernel, axes=0, mode='same'), axis=1)
    return pred


@partial
def size(t: np.ndarray, y: np.ndarray, *, min: int=0, max: int=np.inf) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Specify a minimum and maximum size for a segment.
    Effectively acts as a filter for segment size
    """
    if min < len(t) < max:
        yield t,y


@njit  # jit compiled for speed
def do_lower_cusum(Z, *, omega: float, c: float) -> np.ndarray:
    """
    :param Z: standardized data with mean 0 and S.D. 1
    :param omega: critical level parameter
    :param c: ceiling for the cusum value, helps with dynamic range
    :return: cusum values
    """
    # Pre-allocated numpy array for speed
    S = np.empty(Z.shape)
    S[0] = 0
    for i, (s, z) in enumerate(zip(S[:-1], Z[1:])):
        S[i + 1] = max(0, min(s - z - omega, c))
    return S


def lower_cusum(y, *, mu: float = None, sigma: float = None,
                omega: float = 0, c: float = np.inf) -> np.ndarray:
    """
    Preprocess the data for calculating the lower cusum

    :param y: data
    :param mu: target mean
    :param sigma: target S.D.
    :param omega: tunable critical level parameter
    :param c: optional ceiling to help with dynamic range
    :return: array of cusum control values
    """
    if mu is None:
        mu = np.median(y)
    if sigma is None:
        sigma = np.std(y[~outliers(y)])

    Z = (y - mu) / sigma  # scaling
    return do_lower_cusum(Z, omega=omega, c=c)


@partial
def cusum(t: np.ndarray, y: np.ndarray, *, padding: int=0,
          omega: float, c: float, T: float) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Forward cusum to find start,
    reverse cusum to find end.
    medium events with s.d. ~5% of baseline: omega ~200, c ~1000
    short events with s.d. ~5% baseline: omega 60, c 200

    :param t: time
    :param y: data
    :param mu: target mean
    :param sigma: target S.D.
    :param padding: pad the events by event length * padding
    :param omega: tunable critical level parameter
    :param c: optional ceiling to help with dynamic range
    :param T: threshold
    :yield: event segments
    """
    Sf = lower_cusum(y, omega=omega, c=c)
    Sr = lower_cusum(y[::-1], omega=omega, c=c)[::-1]
    S = (Sf+Sr)/2
    thres = np.diff(S>T,append=0)
    starts = np.arange(len(y))[thres==1]
    ends = np.arange(len(y))[thres==-1]

    for s,e in zip(starts,ends):
        l = e-s
        s = max(0, s-l*padding)
        e = min(len(y)-1, e+l*padding)
        yield t[s:e], y[s:e]


@partial
def split(t: np.ndarray,y: np.ndarray,*,
          maxlen: int) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Split up segments for ease of processing
    :param maxlen: maximum length of a segment
    """
    n_splits = len(t) // maxlen
    if n_splits > 1:
        for t_, y_ in zip(np.array_split(t, n_splits), np.array_split(y, n_splits)):
            yield t_, y_
    else:
        yield t,y


@partial
def switch(t: np.ndarray, y: np.ndarray, threshold: float=0.8) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Segment a raw nanotrace based on manual voltage switch spikes
    using a peak finding algorithm.
    :param threshold: fraction of extrema to consider for peak finding
    """
    hi = np.max(y) * threshold
    lo = np.min(y) * threshold
    his = find_peaks(y, height=hi)[0]
    los = find_peaks(-y, height=-lo)[0]

    # Also add the start and end otherwise we skip segments
    bounds = np.sort(np.concatenate([[0], his, los, [len(y) - 1]]))

    for s, e in zip(bounds[:-1], bounds[1:]):
        yield t[s:e], y[s:e]


@partial
def lowpass(t: np.ndarray, y: np.ndarray, *, cutoff: int,
            abf: ABF, order: int=10) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Wrap a lowpass butterworth filter
    :param t: time
    :param y: data
    :param cutoff: cutoff frequency (Hz)
    :param abf: ABF object
    :param order: filter order (default: 10)
    :yield: filtered data
    """
    sos = signal.butter(order, cutoff, 'lowpass', fs=abf.sampleRate, output='sos')
    filt = signal.sosfilt(sos, y)
    assert len(filt) == len(t)
    yield t, filt


@partial
def as_ires(t: np.ndarray, y: np.ndarray, bl: float | Literal["auto"]='auto', *,
            lo: float=0., hi: float=200., **kwargs) -> Generator[tuple[np.ndarray, np.ndarray] | None]:
    """
    Calculate Ires, optionally using an automatic baseline calculation

    :param t: time
    :param y: data
    :param bl: pre calculated baseline or "auto"
    :parma lo: low bound to consider for baseline calculation
    :param hi: high bound to consider for baseline calculation
    """
    if isinstance(bl, str):
        assert bl == 'auto', "Only 'auto' is accepted as string"
        try:
            bl, _ = baseline(y, lo=lo, hi=hi, **kwargs)
        except BadBaseline as e:
            raise StageError("Could not automatically calculate baseline") from e
    yield t, y / bl


@partial
@wraps(as_ires)
def as_iex(t,y, **kwargs):
    """
    Wrapper for as_ires to calculate Iex instead.
    """
    warnings.warn("At this moment all other stages are written with Ires in mind, take this into consideration.")
    yield t, 1-next(as_ires(**kwargs)(t,y))[1]


@partial
def threshold(t: np.ndarray, y: np.ndarray, *, lo: float=0,
              hi: float, tol: float=0) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Threshold search.
    Segment into consecutive pieces between lo and hi

    :param t: time
    :param y: data
    :param lo: lower bound
    :param hi: upper bound
    :param tol: tolerance factor, defaults to 0.
        Value between 0 and 1
    """
    if tol > 0:
        klen = int((len(y) / 10) * tol)
        kernel = np.full(klen, 1 / klen)
        smooth = fftconvolve(y, kernel, axes=0, mode='same')
    else:
        smooth = y
    mask = (lo < smooth) & (smooth < hi)
    diff = np.diff(mask, prepend=0, append=0)
    start = np.arange(len(diff))[diff == 1]
    end = np.arange(len(diff))[diff == -1]
    for s, e in zip(start, end):
        yield t[s:e], y[s:e]


@partial
def trim(t: np.ndarray, y: np.ndarray, *, left: int=0,
         right: int=1) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Trim off part of the segment

    :param t: time
    :param y: data
    :param left: samples to trim off on the left, can use seconds if multiplied by fs
    :param right: samples to trim off on the right, can use seconds if multiplied by fs
    """
    left = int(left)
    right = int(right)
    yield t[left:-right], y[left:-right]


@partial
def levels(t: np.ndarray, y: np.ndarray, *, fit: None | GaussianMixture=None, n: int=0, tol: float=0,
           sortby: Literal["mean","weight"]='mean') -> Generator[tuple[Sequence, Sequence, Sequence] | None]:
    """
    Detect levels by fitting to a gaussian mixture model with n components.
    tol is a tolerance parameter between 0-1 that smoothens the prediction probabilities
    essentially smoothening out noise in the prediction to get long consecutive levels
    Optionally provide a prefit model

    :param fit: prefitted gaussian mixture model
    :param n: number of components to fit
    :param tol: tolerance parameter
    :param sortby: sort levels based on mean or weight
    """
    # fit a guassian mixture
    if fit is None:
        if n <= 1:
            raise StageError("n needs to be greater than 1")
        try:
            fit = GaussianMixture(n_components=n).fit(y.reshape(-1, 1))
        except ValueError:
            yield [], [], []
            return

    # predict labels for each datapoint
    pred = smooth_pred(y, fit, tol)
    # Get bounds between consecutive segments
    diff = np.diff(pred, append=0)
    bounds = np.arange(len(y))[(diff != 0)]
    bounds = np.concatenate([[0], bounds, [len(y) - 1]])

    # guassian label is pretty random, make pred labels match the sortkey
    if sortby == 'weight':
        sort = np.argsort(-fit.weights_)  # Sorting the negative weights sorts in reverse
    elif sortby == 'mean':
        sort = np.argsort(fit.means_[:, 0])
    else:
        raise ValueError("sortby must be 'mean' or 'weight', not %s" % sortby)
    pred = sort[pred]

    # padded[bounds+1] gives you the label of the segment _following_ the boundary
    padded = np.pad(pred, pad_width=(0, 2), mode='edge')
    for s, e, l in zip(bounds[:-1], bounds[1:], padded[bounds + 1]):
        # l becomes a feature with function name as column name
        yield t[s:e], y[s:e], l

@partial
def volt(t: np.ndarray, y: np.ndarray, *, abf: ABF, v: float) -> Generator[tuple[np.ndarray, np.ndarray] | None]:
    """
    Given the control voltage array and a target voltage,
    cache start and end indices in a closure that slice the sweep at the target voltage.

    :param t: time
    :param y: data
    :param abf: ABF object
    :param v: (float) target voltage
    :return: function that slices the sweep
    """
    c = abf.sweepC
    try:
        start, end = np.arange(len(c))[np.diff(c == v, append=0) != 0]
    except ValueError as e:
        raise StageError("volt not in control voltage array") from e
    yield t[start:end], y[start:end]


@partial
def by_tag(t: np.ndarray, y: np.ndarray, *, abf: ABF,
           pattern: str) -> Generator[tuple[np.ndarray, np.ndarray] | None]:
    """
    Segment a gapfree trace by tags matching a pattern.
    NOTE: must be used as the first stage otherwise the tag times don't make sense.

    :param t: time
    :param y: current
    :param abf: abf file
    :param pattern: regex pattern
    :return: tuple(time,current)
    """
    tags = abf.tagComments
    times = abf.tagTimesSec
    fs = abf.sampleRate

    matching = np.array([i for i, t in enumerate(tags) if re.search(pattern, t) is not None])
    if len(matching) == 0:
        # If the tag is not found, simply don't yield anything
        return

    i = np.asarray([*np.array(times) * fs, -1]).astype(int)
    start = np.arange(len(times))[matching]
    end = start + 1

    for s, e in zip(i[start], i[end]):
        yield t[s:e], y[s:e]

