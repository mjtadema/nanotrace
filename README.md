# Pipeline quickstart guide
`pipeline` is a python library for automated nanopore electrophysiology (1d timeseries) manipulation and feature extraction.

This guide will cover the following topics:

## Table of contents
1. [Installation](#installation)
2. [Usage example](#usage)
3. Available stages
4. [Design of the `Pipeline` class](#pipeline-design)
5. Design of `stages`
6. Design of `extractors`
7. Writing custom `stages`
8. Writing custom `extractors`

## Installation
1. Ask **Matthijs** for an invite to the private github repository (it's private for now as I want to refine it a bit before I publish it).
### Windows & MacOS
2. [Install Anaconda or miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install#windows-installation)
3. Open the Anaconda cmd prompt
4. Run the following command: `pip install git+[link to git]`
### Linux
3. Create a virtual environment where you want to use the pipeline: `$ pip -m venv venv`
4. Activate the virtual environment: `$ . venv/bin/activate`
2. Install the module from the private repo: `$ pip install git+[link to git]`

## Usage
The pipeline is defined and used through the [Pipeline object](#pipeline-design). As a convention, class names use what is known as "CamelCase", while other variables use_this_style_of_naming.

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
fs = abf.SampleRate # get sample rate in Hz
```

The pipeline takes any number of functions (or `callables`) as arguments that make up the stages of the pipeline in the order that they will be run.
We import the `Pipeline` class from the root of the module with `import Pipeline from pipeline`.
We import the pipeline stages using `from pipeline.stages import *`
Available stages can be listed by running `help(pipeline.stages)` or `?pipeline.stages` in iPython or Jupyter notebook.

## Available stages
### Single output segment
- `lowpass(cutoff_fq, fs, order=10)`
    - Apply a lowpass filter with `cutoff_fq` as the cutoff frequency in Hz, `fs` as the sampling rate and `order` as the order of the filter.
    - The sampling rate can be extracted from an abf file using `ABF().SampleRate`
- `as_ires(minsamples=1000)`
    - Calculate the _residual current_ (Ires) from the baseline.
    - Automatically detects the baseline based on a binning approach.
    - `minsamples` determines how many samples a bin needs to be considered a proper level and not just a fast current "spike".
- `trim(left=0, right=1)`
    - Trim off this many samples from the `left` or the `right` side.
    - If the sampling rate was assigned to a variable named `fs`, you can use this to calculate how many _seconds_ to trim off each side using `nseconds * fs`.


### Multiple output segments
- `threshold`
- `levels`
- `switch`

## Pipeline design

