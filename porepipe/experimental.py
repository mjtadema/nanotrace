"""
Experimental pipeline stages not yet ready to be included in the main stages.
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

import numpy as np

try:
    from dtw import dtw
except ImportError:
    dtw = None
from scipy.signal import resample, fftconvolve

from .decorators import partial

@partial
def shapelet(t, y, *, shapelet, include=True, mindist=5, n_resample=1000):
    """
    Try to resample, smoothen and normalize the data as cleanly as possible,
    then find where the shapelet matches the best

    start to give starting index, else ending index
    mindist to consider the shapelet found, can be tuned
    if not found don't yield anything
    """
    # resample the data
    resampled = resample(y, n_resample)

    # norm = normalize(t, resampled, threshold=50)

    # Calculate DTW alignment to find the best fitting subsequence
    aln = dtw(shapelet, y, open_end=True, open_begin=True, step_pattern='asymmetric')
    index = aln.index2

    # Convert index back to original index
    scl = len(y) / len(y)
    index = (index * scl).astype(int)

    print(aln.distance)
    print(index[0], index[-1])

    if aln.distance < mindist:
        if include:
            # include the shapelet itself
            yield t[index[0]:], y[index[0]:]
        else:
            yield t[index[-1]:], y[index[-1]:]

@partial
def normalize(t, y, threshold=0, nbins=5):
    # if threshold is a float between 0 and 1,
    # interpret it as a fraction of the segment
    if 0 <= threshold <= 1:
        threshold = int(len(y) * threshold)

    # Smoothen to get rid of spikes
    wlen = 10
    kernel = np.full(wlen, 1 / wlen)
    smooth = fftconvolve(y, kernel)

    # Discretize the data to find bins with nsamples > threshold to use for normalization
    disc = np.digitize(smooth, np.linspace(0, 1, nbins))
    ind, counts = np.unique(disc, return_counts=True)

    # Find low and high bins
    # find which datapoints are equal to the lowest and highest bins > mincount in one fell swoop
    lo, hi = np.equal(disc[..., None], ind[counts > threshold][[0, -1]][None, ...]).T

    # Normalize the smoothed data based on abundant bins
    normmin = np.min(smooth[lo])
    normmax = np.max(smooth[hi])

    norm = (y - normmin) / (normmax - normmin)
    yield t, norm

