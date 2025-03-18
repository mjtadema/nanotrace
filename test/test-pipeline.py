from pathlib import Path

import pytest
from pipeline import *
from pipeline.stages import *
from pipeline.extractors import freq_by_power, global_features
from functools import partial

@pytest.fixture
def abf_blood():
    path = Path("./test/test_blood.abf")
    return ABF(path)

@pytest.fixture
def pipe_blood(abf_blood):
    fs = abf_blood.sampleRate
    freq = partial(freq_by_power, n=8, fs=fs)
    freq.__name__ = "freq"
    pipe = Pipeline(
        volt(abf_blood.sweepC, 20.0),
        lowpass(cutoff_fq=10e3, fs=fs),
        trim(left=fs * 0.01),
        as_ires(),
        threshold(lo=0.0, hi=0.8, cutoff=1e-3 * fs),
        trim(left=1e-4 * fs, right=1e-4 * fs),
        extractors=(*global_features, freq),
        nsegments=10
    )
    return pipe

def test_blood(pipe_blood, abf_blood):
    assert len(pipe_blood(abf_blood, njobs=4).features) > 0

def test_sublevels():
    abf = ABF("test/test_sublevels.abf")
    fs = abf.sampleRate
    pipe = Pipeline(
        lowpass(cutoff_fq=100, fs=fs),
        as_ires(),
        threshold(lo=0.5, hi=0.6, cutoff=1e-1 * fs, tol=0),
        trim(left=0.1 * fs, right=0.1 * fs),
        levels(n=2),
        extractors=global_features,
        nsegments=10
    )
    pipe(abf).events[0].y = np.array([])
    assert len(pipe(abf).features) > 0