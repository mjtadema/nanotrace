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

from .exception import StageError
from .decorators import partial, cutoff
# from .src import cusum as _do_cusum
def _do_cusum(): pass



# Utilities
def baseline(y, minsamples, max_amplitude=500) -> float:
    """
    Automatic baseline calculation
    :param y: current array
    :param minsamples: minimum number of samples that must be in the bin
    :param max_amplitude: maximum amplitude to consider as baseline
    :return: baseline
    """
    # Divide data into bins, with log spacing
    # Get rid of the polarity in the calculation
    nbins = 20
    counts, edges = np.histogram(np.abs(y), bins=(np.logspace(0, 3, nbins)))
    bins = np.array([a+b/2 for a,b in zip(edges[:-1], edges[1:])])
    # Digitize based on the same bins
    digi = np.digitize(np.abs(y), bins=edges)
    # Determine highest bin over a certain threshold of samples
    # to get rid of spikes
    i = np.arange(len(counts))[(counts > minsamples) & (bins < max_amplitude)][-1] + 1
    return np.median(y[digi == i])


def smooth_pred(y, fit_, tol):
    """
    Smoothen a gaussian mixture prediction
    """
    # tol between 0 and 1?

    if tol <= 0:
        # Special case, do the regual prediction
        return fit_.predict(y.reshape(-1, 1))
    elif tol > 1:
        tol = 1
    proba = fit_.predict_proba(y.reshape(-1, 1))
    klen = int((len(y) / 10) * tol)
    kernel = np.full((klen, proba.shape[1]), 1 / klen)
    pred = np.argmax(fftconvolve(proba, kernel, axes=0, mode='same'), axis=1)
    return pred


@partial
@cutoff
def cusum(t, y, *, w: float, T: float):
    """
    Use a CUSUM based method to detect continuous segments above T (threshold)
    :param w: weight
    :param T: threshold
    """
    Z = y / np.std(y)
    S = np.empty(Z.shape, dtype=np.float32)
    for i in range(1, len(Z)):
        x = S[i - 1]
        z = Z[i]
        S[i] = x - z - w
    mask = S > T
    diff = np.diff(mask, prepend=0, append=0)
    start = np.arange(len(diff))[diff == 1]
    end = np.arange(len(diff))[diff == -1]
    for s, e in zip(start, end):
        yield t[s:e], y[s:e]


@partial #Resample the data such that the time resolution is now "tres" points per second.
def resample(t, y, *, fs, TimeRes):
    ResamplingFactor = fs / TimeRes
    if ResamplingFactor <= 1:
        print("Warning: Your new time resolution should be smaller than the current one!")
    ResampledLength = int(len(t)/ResamplingFactor)
    downsampling = signal.resample(y, ResampledLength, t)
    #print(downsampling[1])
    yield downsampling[1], downsampling[0]


@partial
@cutoff
def switch(t, y):
    """
    Segment a raw nanotrace based on manual voltage switch spikes
    """
    hi = np.max(y) / 1.2
    lo = np.min(y) / 1.2
    his = find_peaks(y, height=hi)[0]
    los = find_peaks(-y, height=-lo)[0]

    # Also add the start and end otherwise we skip segments
    bounds = np.sort(np.concatenate([[0], his, los, [len(y) - 1]]))

    for s, e in zip(bounds[:-1], bounds[1:]):
        yield t[s:e], y[s:e]


@partial
def lowpass(t, y, *, cutoff_fq: int, abf: ABF, order: int=10):
    """
    Wrap a lowpass butterworth filter
    :param t:
    :param y:
    :param cutoff_fq:
    :param fs:
    :param order:
    :return:
    """
    sos = signal.butter(order, cutoff_fq, 'lowpass', fs=abf.sampleRate, output='sos')
    filt = signal.sosfilt(sos, y)
    assert len(filt) == len(t)
    yield t, filt


@partial
def as_ires(t, y, max_amplitude: int=200, minsamples: int=1000):
    """Calculate Ires using an automatic baseline calculation"""
    yield t, y / baseline(y, minsamples, max_amplitude=max_amplitude)


@partial
@cutoff
def threshold(t, y, *, lo: float, hi: float, tol: float=0):
    """
    Segment into consecutive pieces between lo and hi
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
def trim(t, y, *, left: int=0, right: int=1):
    """
    Trim off part of the segment
    :param left: samples to trim off on the left, can use seconds if multiplied by fs
    :param right: samples to trim off on the right, can use seconds if multiplied by fs
    """
    left = int(left)
    right = int(right)
    yield t[left:-right], y[left:-right]


@partial
@cutoff
def levels(t, y, *, n: int, tol: float=0, sortby: str='mean'):
    """
    Detect levels by fitting to a gaussian mixture model with n components.
    tol is a tolerance parameter between 0-1 that smoothens the prediction probabilities
    essentially smoothening out noise in the prediction to get long consecutive levels
    """

    # fit a guassian mixture
    try:
        fit_ = GaussianMixture(n_components=n).fit(y.reshape(-1, 1))
    except ValueError:
        yield [], [], []
        return
    # predict labels for each datapoint
    pred = smooth_pred(y, fit_, tol)
    # Get bounds between consecutive segments
    diff = np.diff(pred, append=0)
    bounds = np.arange(len(y))[(diff != 0)]
    bounds = np.concatenate([[0], bounds, [len(y) - 1]])

    # guassian label is pretty random, make pred labels match the sortkey
    if sortby == 'weight':
        sort = np.argsort(-fit_.weights_)  # Sorting the negative weights sorts in reverse
    elif sortby == 'mean':
        sort = np.argsort(fit_.means_[:, 0])
    else:
        raise ValueError("sortby must be 'mean' or 'weight', not %s" % sortby)
    pred = sort[pred]

    # padded[bounds+1] gives you the label of the segment _following_ the boundary
    padded = np.pad(pred, pad_width=(0, 2), mode='edge')
    for s, e, l in zip(bounds[:-1], bounds[1:], padded[bounds + 1]):
        # l becomes a feature with function name as column name
        yield t[s:e], y[
                      s:e], l  # This is the way to smuggle out extra information without having access to the segment yet

@partial
def volt(t,y,*,abf: ABF,v: float):
    """
    Given the control voltage array and a target voltage,
    cache start and end indices in a closure that slice the sweep at the target voltage.

    :param c: (array) control voltage
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
def by_tag(t: np.ndarray, y: np.ndarray, *, abf: ABF, pattern: str):
    """
    Segment a gapfree nanotrace by tags matching a pattern.
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

