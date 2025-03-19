# Pipeline

`pipeline` is a python library for automated nanopore electrophysiology (1d timeseries) manipulation and feature extraction.

This guide will cover the following topics:

## Table of contents
1. [Installation](#installation)
2. [Updating](#updating)
3. [Usage example](#usage)
4. [Available stages](#available-stages)
    1. [Custom stages](#defining-a-custom-stage)

## Installation
1. Ask **Matthijs** for an invite to the private github repository (it's private for now as I want to refine it a bit before I publish it).

### Windows
2. [Install GIT](https://git-scm.com/downloads/win)
3. [Install Anaconda or miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install#windows-installation)
4. Create a virtual environment that you want to use for 
5. Open the Anaconda cmd prompt **from the correct environment**
6. Run the following command: `pip install git+https://github.com/mjtadema/pipeline.git`

### Linux
2. Install git from whatever software repository you use (i.e. `sudo apt install git` for ubuntu/debian)
3. (optional) install conda if you prefer
4. Create a virtual environment where you want to use the pipeline: `$ pip -m venv venv`
5. Activate the virtual environment: `$ . venv/bin/activate`

### All platforms
7. Install the module from the private repo: `pip install git+https://github.com/mjtadema/pipeline.git`

## Updating
Update using the latest development version **(Recommended)**: `pip install --upgrade --force-reinstall --no-deps git+https://github.com/mjtadema/pipeline.git`

Update using the latest stable version: `pip install --upgrade --force-reinstall --no-deps git+https://github.com/mjtadema/pipeline.git@master`

## Usage
The pipeline is defined and used through the [Pipeline object](#pipeline-design). As a convention, class names use what is known as "CamelCase", while other variables use_this_style_of_naming. Available pipeline stages can be found [here](#available-stages).

### Pipeline definition
```python
# Example:
import Pipeline from pipeline
from pipeline.stages import *
import pipeline

help(pipeline.stages)

pipeline = Pipeline(
    stage_1,
    stage_2,
    stage_3
)

import ABF from pyabf

abf = ABF("some_abf_file.abf")
fs = abf.sampleRate # get sample rate in Hz
```

The pipeline takes any number of functions (or `callables`) as arguments that make up the stages of the pipeline in the order that they will be run.
We import the `Pipeline` class from the root of the module with `import Pipeline from pipeline`.
We import the pipeline stages using `from pipeline.stages import *`
Available stages can be listed by running `help(pipeline.stages)` or `?pipeline.stages` in iPython or Jupyter notebook.

## Available stages
### Single output segment

| Syntax                             | Description                                                                                                                                                                                                                                          |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lowpass(cutoff_fq, fs, order=10)` | Apply a lowpass filter with `cutoff_fq` as the cutoff frequency in Hz, `fs` as the sampling rate and `order` as the order of the filter. The sampling rate can be extracted from an abf file using `ABF().SampleRate`                                |
| `as_ires(minsamples=1000)`         | Calculate the _residual current_ (Ires) from the baseline. Automatically detects the baseline based on a binning approach. `minsamples` determines how many samples a bin needs to be considered a proper level and not just a fast current "spike". |
| `trim(left=0, right=1)`            | Trim off this many samples from the `left` or the `right` side.  If the sampling rate was assigned to a variable named `fs`, you can use this to calculate how many _seconds_ to trim off each side using `nseconds * fs`.                           |

### Multiple output segments

| Syntax                                            | Description                                                                                                                                                                                                                                                                                                                                                                                                           |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `switch()`                                        | Segment a gapfree trace based on large, short, current spikes cause by manual voltage switching.                                                                                                                                                                                                                                                                                                                      |
| `threshold(lo,hi)`                                | Segment an input segment by consecutive stretches of current between `lo` and `hi`.                                                                                                                                                                                                                                                                                                                                   |
| `levels(n, tol=0, sortby='mean')`                 | Detect sublevels by fitting a [gaussian mixture model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). Use `n` to set the number of gaussians to fit, `tol` is a number between 0 and 1 and controls how much short spikes are tolerated. `sortby` controls how the gaussians are labeled, can be sorted by "mean" or by "weight" (weight being the height of the gaussian). |

### Decorators
[Decorators](https://peps.python.org/pep-0318/) are functions that wrap around other functions with a convenient syntax. I use them to _enhance_ the "default" behavior of the stages and they live in `pipeline.decorators`. The following decorators are predefined:

| Name                                | Description                                                                                                                                                                                                                                                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cutoff`                           | Add a filter to a stage that blocks yielding segments _smaller than_ the cutoff. `cutoff` adds the "cutoff" keyword argument to a stage.                                                                                                                                                                                      |
| `partial`                          | Essentially functions as [functools.partial](https://docs.python.org/3/library/functools.html) but in a decorator form for convenience. Allows pre-defining some arguments when the decorated function is called. I use it to set keyword arguments and only leave _positional arguments_ to be filled when the stage is run. |


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
from pipeline.decorators import partial

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

The `cutoff` decorator is used on a many built-in stages to filter out segments that are too short. It can be added to a custom stage like so:

```python
from pipeline.decorators import cutoff

@cutoff
def new_stage(t,y):
    """An example pipeline stage that "yields" new segments"""
    t_segments = f(t)
    y_segments = f(y)
    for new_t, new_y in zip(t_segments, y_segments):
        # Using "yield" turns the function into a generator
        yield new_t, new_y
```

## Extractors
**coming soon**
