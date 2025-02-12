import numpy as np
import pandas as pd
from anytree import RenderTree
from scipy.signal import fftconvolve

from pathlib import Path
from typing import Union

from pyabf import ABF

ABFLike = Union[ABF, str, Path]

def as_abf(abf: ABFLike) -> ABF:
    if not isinstance(abf, ABFLike):
        raise TypeError(('Expected an AbfLike, not type', type(abf)))
    if isinstance(abf, str):
        abf = Path(abf)
    if isinstance(abf, Path):
        if not abf.exists(): raise FileNotFoundError(abf)
        abf = ABF(abf)
    return abf


def polarity(y):
    return np.sign(np.median(y))


def baseline(y, minsamples):
    # Divide data into bins, with log spacing
    # Get rid of the polarity in the calculation
    nbins = 20
    counts, bins = np.histogram(y * polarity(y), bins=(np.logspace(0, 3, nbins)))
    # Digitize based on the same bins
    digi = np.digitize(y * polarity(y), bins=bins)
    # Determine highest bin over a certain threshold of samples
    # to get rid of spikes
    i = np.arange(len(counts))[counts > minsamples][-1] + 1
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


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


class PoolMixin:
    """
    Add pool method to NodeMixin
    """
    def pool(self):
        """
        Pool features from children
        """
        if self.children:
            return pd.concat([child.features for child in self.children], ignore_index=True)
        else:
            return pd.DataFrame()

class ReprMixin:
    """
    Add tree rendering
    """
    def __repr__(self):
        """Fancy tree rendering"""
        out = []
        render = iter(RenderTree(self))
        prev = None
        for pre, _, node in render:
            cnt = 0
            while prev == pre:
                # skip until we encounter next level
                pre, _, node = next(render)
                cnt += 1
            else:
                if cnt > 0:
                    out.append("%s ... Skipped %d segments" % (prev, cnt))
            out.append("%s%s" % (pre, str(node)))
            prev = pre
        return '\n'.join(out)