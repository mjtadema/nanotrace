from __future__ import annotations
from pathlib import Path

import pytest
from porepipe import *
from porepipe.stages import *
from porepipe.features import psd_freq, global_features
from functools import partial

@pytest.fixture
def abf_blood():
    path = Path("./test/test_blood.abf")
    return ABF(path)

@pytest.fixture
def pipe_blood(abf_blood):
    fs = abf_blood.sampleRate
    pipe = Pipeline(
        volt(abf=abf_blood, v=20.0),
        lowpass(cutoff_fq=10e3, abf=abf_blood),
        trim(left=fs * 0.01),
        as_ires(),
        threshold(lo=0.0, hi=0.8, cutoff=1e-3 * fs),
        trim(left=1e-4 * fs, right=1e-4 * fs),
        features=(*global_features, psd_freq(fs=fs)),
        n_segments=10,
        n_jobs=4
    )
    return pipe

def test_blood(pipe_blood, abf_blood):
    assert len(pipe_blood(abf_blood).features) > 0
    print(pipe_blood)
    print(pipe_blood(abf_blood))
    pipe_blood(abf_blood).by_name['volt'][0].inspect()

def test_sublevels():
    abf = ABF("test/test_sublevels.abf")
    fs = abf.sampleRate
    pipe = Pipeline(
        lowpass(cutoff_fq=100, abf=abf),
        as_ires(max_amplitude=150),
        threshold(lo=0.55, hi=0.7, cutoff=2 * fs),
        trim(left=0.01 * fs, right=0.01 * fs),
        levels(n=2, tol=0.05),
        features=global_features,
        n_segments=10,
        n_jobs=4
    )
    pipe(abf).events[0].y = np.array([])
    assert len(pipe(abf).features) > 0