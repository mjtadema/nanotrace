# Import main Pipeline class and ABF class
from nanotrace import Pipeline, ABF
# Import pipeline stages
from nanotrace.stages import volt, baseline_from_sweeps, as_ires, lowpass, cusum, size, trim
# Import feature extractors
from nanotrace.features import global_features # List of default features

# Load the ABF file
abf = ABF("test/test_protein.abf")
# Sample rate is used to easily use time in calculations instead of number of samples
fs = abf.sampleRate
# Get the median baseline and standard deviation from all sweeps between -100 and -160
bl, sd = baseline_from_sweeps(abf, lo=-160, hi=-100)

# Setup the pipeline stages
pipe = Pipeline(
    # Take only the part of the trace at -100mV
    volt(abf=abf, v=-100),
    # Apply a lowpass butterworth filter at 10kHz
    lowpass(cutoff=10e3, abf=abf),
    # Trim off a part from the left side of the trace where the circuit is not fully charged
    trim(left=0.05 * fs),
    # Calculate residual curent (Ires) based on the baseline in bl
    as_ires(bl=bl),
    # Use a modified CUSUM method to detect events
    cusum(mu=1, sigma=sd / bl, omega=200, c=1000),
    # Filter events between a minimum and maximum length
    size(min=1e-4 * fs, max=1e-1 * fs),
    # Extract global features (mean ires, standard deviation, log dwelltime, median ires,
    #   skewness, kurtosis, high and low means of gaussian mixture model)
    features=global_features,
    # Use at most 4 processors to parallelize feature extraction
    n_jobs=4
)

# Inspect the events downstream of the "as_ires" stage
pipe(abf).inspect('as_ires')

# Plot the features "mean ires" and "log dwelltime" as a scatter plot
pipe(abf).features.plot('mean','ldt','scatter')