from __future__ import annotations

from nanotrace import Pipeline, ABF
from nanotrace.stages import volt, baseline_from_sweeps, as_ires, lowpass, cusum, size, trim
from nanotrace.features import global_features

def test_pipeline():
    """
    Data taken from: https://doi.org/10.5281/zenodo.8123395
    """
    abf = ABF("test/test_protein.abf")
    fs = abf.sampleRate
    bl, sd = baseline_from_sweeps(abf, lo=-160, hi=-100)
    pipe = Pipeline(
        volt(abf=abf, v=-100),
        lowpass(cutoff=10e3, abf=abf),
        trim(left=0.05 * fs),
        as_ires(bl=bl),
        cusum(mu=1, sigma=sd / bl, omega=200, c=1000),
        size(min=1e-4 * fs, max=1e-1 * fs),
        features=global_features,
        n_segments=10,
        n_jobs=4
    )
    features = pipe(abf).features
    assert len(features) == 80