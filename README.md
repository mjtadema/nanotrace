![nanotrace logo](https://github.com/mjtadema/nanotrace/blob/master/figures/trace_logo.png?raw=true)

[![DOI](https://zenodo.org/badge/931659102.svg)](https://doi.org/10.5281/zenodo.15731069) [![PyPI version](https://badge.fury.io/py/nanotrace.svg)](https://badge.fury.io/py/nanotrace) [![Test Python package](https://github.com/mjtadema/nanotrace/actions/workflows/test_pipeline.yml/badge.svg)](https://github.com/mjtadema/nanotrace/actions/workflows/test_pipeline.yml)

# NanoTRACE — Nanopore Toolkit for Reproducible Analysis of Conductive Events
`nanotrace` is a python library for automated nanopore electrophysiology (1d timeseries) manipulation and feature extraction.
In short, it uses a tree based datastructure (based on the [anytree](https://github.com/c0fec0de/anytree) project), to intuitively handle a linear sequence of operations (pipeline stages).
These operations are composed of simple callables that can either return several segments or return 1 or no segments, acting as a filter.
The segments that reach the leaves of the tree are interally called "events" and are the end product of the pipeline.
Features can be defined as a set of callables that compute a feature metric from a single segment.
For parallelization, [joblib](https://github.com/joblib/joblib) is used to support a variety of multiprocessing/threading backends for feature extraction.

## Graphical abstract
![graphical abstract](https://github.com/mjtadema/nanotrace/blob/master/figures/abstract.png?raw=true)

## Table of contents
This guide covers the following topics:

1. [Installation](#installation)
2. [Updating](#updating)
3. [Usage example](#usage)
4. [Available stages](#available-stages)
    1. [Custom stages](#defining-a-custom-stage)
5. [Inspection and validation](#inspection-and-validation)
   1. [Example](#example)
6. [Feature extraction](#feature-extraction)
   1. [Example](#example-1)
7. [Compound pipes](#compound-pipes)
   1. [Example](#example-2)

## Installation
Install the latest release from PyPi: `pip install nanotrace`

## Usage
The pipeline is defined and used through the [Pipeline object](#pipeline-design). As a convention, class names use what is known as "CamelCase", while other variables use_this_style_of_naming. Available pipeline stages can be found [here](#available-stages).

### Pipeline definition

```python
# Example:
from nanotrace import *
# This imports Pipeline, ABF, stages and feature extractors

# run `help(nanotrace.stages)` to list built-in stages
# run `help(nanotrace.features)` to list built-in feature extractors

# Defining the ABF object separately is handy because 
# we often need access to the sample rate
abf = ABF("some_abf_file.abf")
fs = abf.sampleRate # get sample rate in Hz

# Define the pipeline with some stages
pipeline = Pipeline(
    stage_1(),
    stage_2(),
    stage_3()
)
```

The pipeline takes any number of functions (or `callables`) as arguments that make up the stages of the pipeline in the order that they will be run.
You can also import the `Pipeline` class from the root of the module with `from Pipeline import pipeline`.
You can also import the pipeline stages using `from porepipe.stages import *`
Available stages can be listed by running `help(pipeline.stages)` or `?pipeline.stages` in iPython or Jupyter notebook.

## Available stages
### Filters
| Syntax | Description
|--------|-----------
|`size(min, max)` | Specify a minimum (`min`) and maximum (`max`) segment size in terms of number of samples. Segments that fall outside of this range are not passed through to any downstream stages.


### Single output segment

| Syntax                              | Description                                                                                                                                                                                                                                                                                    |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lowpass(cutoff, abf, order=10)`    | Apply a lowpass filter with `cutoff` as the cutoff frequency in Hz, `abf` as the abf file to use as a reference (for sampling rate etc.) and `order` as the order of the filter (unlikely to need to be changed). |
| `as_ires(lo, hi, min_samples=1000)` | Calculate the _residual current_ (Ires) from the baseline. Automatically detects the baseline based on a binning approach. `min_samples` determines how many samples a bin needs to be considered a proper level and not just a fast current "spike".                                          |
| `as_iex(lo, hi, min_samples=1000)`  | Same as `as_ires` but calculate _excluded current_ (Iex). **NOTE:** all other stages are written with Ires in mind for now so take that into consideration.                                                                                                                                                                                                                                      
| `trim(left=0, right=1)`             | Trim off this many samples from the `left` or the `right` side.  If the sampling rate was assigned to a variable named `fs`, you can use this to calculate how many _seconds_ to trim off each side using `nseconds * fs`.                                                                     |

### Multiple output segments

| Syntax                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                           |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `switch(threshold=0.8)`                      | Segment a gapfree trace based on large, short, current spikes caused by manual voltage switching using a peak finding algorithm. `threshold` is the fraction of the extrema to consider when finding peaks                                                                                                                                                                                                            |
| `threshold(lo, hi, tol=0)`                   | Segment an input segment by consecutive stretches of current between `lo` and `hi`. `tol` is a tolerance parameter to tolerate small excursions outside of the threshold (should be a small value like 0.001 - 0.01).                                                                                                                                                                                                 |
| `levels(n, tol=0, sortby='mean')`            | Detect sublevels by fitting a [gaussian mixture model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). Use `n` to set the number of gaussians to fit, `tol` is a number between 0 and 1 and controls how much short spikes are tolerated. `sortby` controls how the gaussians are labeled, can be sorted by "mean" or by "weight" (weight being the height of the gaussian). |
| `volt(abf, v)`                               | Select part of a sweep where the control voltage in `abf` matches the target voltage `v`                                                                                                                                                                                                                                                                                                                              |
| `by_tag(abf, pattern)`                       | Segment a trace into smaller pieces based on matches with `pattern`. `pattern` can be any regex pattern.                                                                                                                                                                                                                                                                                                              
| `cusum(mu, sigma, omega, c, padding: int=0)` | Event detection using CUSUM method. `mu` is the target mean, `sigma` is the standard deviation around the mean, `omega` is the tunable critical level parameter, `c` is the ceiling for the CUSUM control value.                                                                                                                                                                                                      
### Decorators
[Decorators](https://peps.python.org/pep-0318/) are functions that wrap around other functions with a convenient syntax. I use them to _enhance_ the "default" behavior of the stages and they live in `porepipe.decorators`. The following decorators are predefined:

| Name                                | Description                                                                                                                                                                                                                                                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `partial`                          | Essentially functions as [functools.partial](https://docs.python.org/3/library/functools.html) but in a decorator form for convenience. Allows pre-defining some arguments when the decorated function is called. I use it to set keyword arguments and only leave _positional arguments_ to be filled when the stage is run. |
| `catch_errors(n=1)` | Used for feature calculation. Catches errors and simply returns the specified number of NaN values to not interrupt the feature calculation. |


### Defining a custom stage
Stages are functions (`callable`s) that take only two _positional arguments_: `t`(time) and `y`(current). The function then does something to transform the data or calculate new segments and `yield`s segments. By using `yield` instead of `return` the function is turned into a [generator](https://docs.python.org/3/reference/expressions.html#yieldexpr) and can be used as an `iterable`. All stages need to be `generator`s or return an `iterable`.

```python
def new_stage(t,y):
    """An example pipeline stage that "yields" new segments"""
    t_segments = f(t)
    y_segments = f(y)
    for new_t, new_y in zip(t_segments, y_segments):
        # Using "yield" turns the function into a generator
        yield new_t, new_y
```

The stage can then be given to the pipeline like so:

```python
Pipeline(
    new_stage
)
```

Extra options can be given when the pipeline is defined by using the `partial` decorator when defining the function like so:

```python
from nanotrace.decorators import partial

@partial
def new_stage(t,y,*,extra_argument):
    """An example pipeline stage that "yields" new segments"""
    t_segments = f(t, extra_argument)
    y_segments = f(y, extra_argument)
    for new_t, new_y in zip(t_segments, y_segments):
        # Using "yield" turns the function into a generator
        yield new_t, new_y

Pipeline(
    new_stage(extra_argument)
)
```

## Inspection and validation
The main advantage of using a `Tree` datastructure is that every segment generated by the pipeline is connected to its parent segment. This means that along every step of the pipeline, the stage input and output can be plotted and inspected to ensure the output matches expectations. To aid in this there are a couple of convenience functions:

`Segment.plot` is a thin wrapper around `matplotlib.pyplot.plot` to make it easy to plot segment data. It implements one additional keyword argument of its own called `normalize`. This removes the time from the plot and instead generates new `x` values using `np.linspace` between 0 and 1. The effect is that all events get plotted on top of each other. Keyword arguments meant for `matplotlib.pyplot.plot` get passed through as expected. 

`Segment.inspect` is a convenience function that plots events (lowest level segments) on top of itself. This way you get an overview of the effect of all the stages downstream of the stage that `inspect` was called on.

`Root.inspect` is a convenience function to call `inspect` on a named step and provides an interactive plot that allows scrolling through all segments of that level.
### Example:

```python
from nanotrace import *

abf = ABF("some_abf_file.abf")
fs = abf.sampleRate

pipe = Pipeline(
    volt(abf.sweepC, 20),
    lowpass(cutoff=10e3, abf=abf),
    trim(left=0.01*fs),
    as_ires(),
    threshold(lo=0.0, hi=0.8),
    size(min=1e-3*fs),
    features=(mean, ldt),
    n_segments=10,
    n_jobs=4
)
pipe(abf).by_name['as_ires'][0].inspect()
```
![inspect example output](https://github.com/mjtadema/nanotrace/blob/master/figures/inspect.png?raw=true)

## Feature extraction
After segmenting a trace and detecting events, features can be extracted. This generally means that a single event gets reduced to several characteristic quantities that we call _features_, such as the mean current value (using `mean`) or the dwell-time (using `dt`), among other features. Below is a working `Pipeline` definition with feature extraction to extract the mean current and the dwelltime from the events resulting from the `Pipeline`.

As the features are kept in a standard `pandas.DataFrame`, the standard [pandas convenience plotting methods](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html) can be used for plotting. 
A custom plotting function is added to the pandas plot wrapper for convience and can be accessed from `pandas.DataFrame.plot.dens2d()`.
It takes two column names and plots them as a scatter plot where the markers are colored by the density of the datapoints.

### Example:

```python
from nanotrace import *

abf = ABF("some_abf_file.abf")
fs = abf.sampleRate

pipe = Pipeline(
    volt(abf.sweepC, 20),
    lowpass(cutoff=10e3, abf=abf),
    trim(left=0.01*fs),
    as_ires(),
    threshold(lo=0.0, hi=0.8),
    size(min=1e-3*fs),
    features=(mean, ldt),
    n_segments=10,
    n_jobs=4
)
pipe(abf).features.plot('mean','ldt','scatter')
```
![features example output](https://github.com/mjtadema/nanotrace/blob/master/figures/features.png?raw=true)


## Compound pipes
Pipelines can be added together using the `|` operator (in unix terms also known as a pipe).

### Example:
```python
from nanotrace import *

abf = ABF("some_abf_file.abf")
fs = abf.sampleRate

first = Pipeline(
    volt(abf.sweepC, 20),
    lowpass(cutoff=10e3, abf=abf),
    trim(left=0.01*fs),
    as_ires(),
)

# here we could do some calculations based on the first part of the pipe
# and use this for the second pipe definition

second = Pipeline(
    threshold(lo=0.0, hi=0.8),
    size(min=1e-3*fs),
    features=(mean, ldt),
    n_segments=10,
    n_jobs=4
)

pipe = first | second
```
