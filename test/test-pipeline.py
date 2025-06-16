from __future__ import annotations
from pathlib import Path

import pytest
import numpy as np

from nanotrace import Pipeline, ABF
from nanotrace.stages import (
    volt,
    lowpass,
    trim,
    as_ires,
    threshold,
    levels,
    baseline_from_sweeps,
    cusum,
    size
)
from nanotrace.features import (
    global_features,
    psd_freq,
    peptide_fit
)

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

def test_peptides():
    abf = ABF("test/test_peptides.abf")
    fs = abf.sampleRate
    bl, sd = baseline_from_sweeps(abf, min_amplitude=30, max_amplitude=100)
    first = Pipeline(
        # by_tag(abf=abf, pattern=f" {select.iloc[i].v} mV"), # find segments of the trace where the tag matches the pattern (150mV)
        volt(abf=abf, v=150),
        trim(left=0.5 * fs, right=0.5 * fs),  # trim off the ends of the segments where the current ramps up or down
        lowpass(cutoff_fq=10e3, abf=abf),  # pass the segment through a lowpass filter at 10kHz
        as_ires(bl=bl),  # detect the baseline between the min and max amplitudes and calculate Ires
    )
    second = Pipeline(
        cusum(mu=1, sigma=sd / bl, omega=60, c=200),  # event detection using cusum method
        size(min=1e-4 * fs, max=1 * fs),  # filter events by size
        features=[peptide_fit],
        n_segments=30,
        n_jobs=4
    )
    pipe = first | second
    assert len(pipe(abf).features) > 0