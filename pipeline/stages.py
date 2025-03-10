"""
Functions to be used as refinement steps in a pipeline
These must take time and current arrays as arguments
and return (or yield) an iterator of "refined" time and current arrays
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

from .utils import baseline, smooth_pred
from .decorators import partial, cutoff


@cutoff
def switch(t,y):
    """
    Segment a raw trace based on manual voltage switch spikes
    """
    hi = np.max(y) / 1.2
    lo = np.min(y) / 1.2
    his = find_peaks(y, height=hi)[0]
    los = find_peaks(-y, height=-lo)[0]

    # Also add the start and end otherwise we skip segments
    bounds = np.sort(np.concatenate([[0], his, los, [len(y)-1]]))

    for s, e in zip(bounds[:-1], bounds[1:]):
        yield t[s:e], y[s:e]


@partial
def lowpass(t,y, *, cutoff_fq, fs, order=10):
    """Wrap a lowpass butterworth filter"""
    sos = signal.butter(order, cutoff_fq, 'lowpass', fs=fs, output='sos')
    filt = signal.sosfilt(sos, y)
    assert len(filt) == len(t)
    yield t, filt


@partial
def as_ires(t,y,minsamples=1000):
    """Calculate Ires using an automatic baseline calculation"""
    yield t, y/baseline(y,minsamples)


@partial
@cutoff
def binned(t, y, *, lo=0, hi=1, nbins=5):
    """yield segments as sequential bins"""
    bins = np.linspace(lo, hi, nbins)
    digi = np.digitize(y, bins)
    diff = np.diff(digi, append=0)
    bounds = np.arange(len(y))[(diff != 0)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        Y = y[s:e]
        if lo < np.median(Y) < hi:
            yield t[s:e], Y


@partial
@cutoff
def threshold(t,y,*,lo,hi):
    """
    Segment into consecutive pieces between lo and hi
    """
    mask = (lo < y) & (y < hi)
    diff = np.diff(mask, prepend=0, append=0)
    start = np.arange(len(diff))[diff == 1]
    end = np.arange(len(diff))[diff == -1]
    for s,e in zip(start,end):
        yield t[s:e], y[s:e]


@partial
def trim(t,y,*,left=0,right=1):
    """Trim off part of the segment"""
    left = int(left)
    right = int(right)
    yield t[left:-right], y[left:-right]


@partial
@cutoff
def levels(t, y, *, n, tol=0, sortby='mean'):
    """
    Detect levels by fitting to a gaussian mixture model with n components.
    tol is a tolerance parameter between 0-1 that smoothens the prediction probabilities
    essentially smoothening out noise in the prediction to get long consecutive levels
    """

    # fit a guassian mixture
    fit_ = GaussianMixture(n_components=n).fit(y.reshape(-1, 1))
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
        yield t[s:e], y[s:e], l  # This is the way to smuggle out extra information without having access to the segment yet
