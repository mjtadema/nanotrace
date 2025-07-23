from nanotrace import Pipeline, ABF
from nanotrace.stages import volt, baseline_from_sweeps, as_ires, lowpass, cusum, size, trim
from nanotrace.features import global_features, sequence_features

def test_protein():
    """
    Data taken from: https://doi.org/10.5281/zenodo.8123395
    """
    abf = ABF("test/test_protein.abf")
    fs = abf.sampleRate
    pipe = Pipeline(
        volt(abf=abf, v=-100),
        lowpass(cutoff=10e3, abf=abf),
        trim(left=0.05 * fs),
        as_ires(lo=-160, hi=-100),
        cusum(omega=10, c=2, T=1),
        size(min=1e-4 * fs, max=1e-1 * fs),
        features=global_features,
        n_segments=10,
        n_jobs=4
    )
    features = pipe(abf).features
    assert len(features) == 61