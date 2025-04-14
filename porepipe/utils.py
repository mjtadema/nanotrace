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

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from anytree import RenderTree
from pyabf import ABF
from scipy.signal import fftconvolve

ABFLike = Union[ABF, str, Path]
ABFLikeTypes = [ABF, str, Path]


def as_abf(abf: ABFLike) -> ABF:
    if not type(abf) in ABFLikeTypes:
        raise TypeError(('Expected an AbfLike, not type', type(abf)))
    if isinstance(abf, str):
        abf = Path(abf)
    if isinstance(abf, Path):
        if not abf.exists(): raise FileNotFoundError(abf)
        abf = ABF(abf)
    return abf


def polarity(y):
    """Calculate the polarity of a trace by determining the sign of the median current"""
    return np.sign(np.median(y))


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


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return data[s < m]


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
