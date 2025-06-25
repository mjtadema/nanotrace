from __future__ import annotations

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

import logging
from pathlib import Path
from typing import Sequence, Callable

from .exception import NanotraceException
from . import segment
from .abf import ABFLike, as_abf, AbfRoot, ABFLikeTypes

logger = logging.getLogger(__name__)


class Pipeline:
    """
    The Pipeline class is the main class of this module.
    Its job is to define the pipeline through _stages_, functions that each modify timeseries data
    as steps in a pipeline.

    Example:
        ```
        ref = Pipeline(
            slices(slices=slice_list),
            lowpass(cutoff_fq=10000, fs=fs),
            as_ires(),
            threshold(lo=0.4, hi=0.65, cutoff=0.001*fs),
        )
        ```
    """

    def __init__(self, *stages: Sequence[Callable], n_jobs: int=1,
                 features: tuple | None=None, **kwargs) -> None:
        """
        A pipeline is constructed as a linear list of pipeline "stages".

        :param stages: a list of stages (callables) that make up the pipeline steps
        :param kwargs: additional keyword arguments are passed to the root segment
        """
        # The pipeline instance caches the root segment with the abf file paths as keys
        self._cache = {}
        logger.debug("Constructing pipeline with %d steps: %s", len(stages), ",".join([f.__name__ for f in stages]))
        self.stages = list(stages)
        self.features = features if features is not None else []
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def __str__(self) -> str:
        repr = "Pipeline with %d stage(s): " % (len(self.stages))
        repr += ', '.join([stage.__name__ for stage in self.stages])
        return repr

    def __repr__(self) -> str:
        return str(self)

    def __or__(self, other: "Pipeline") -> "Pipeline":
        """
        "pipe" together two pipelines
        merge stages and feature extractors
        overwrite other settings such as n_jobs
        and other kwargs from the second pipe.
        """
        self._cache = {} # reset cache
        self.stages.extend(other.stages)
        self.features.extend(other.features)
        self.n_jobs = other.n_jobs
        self.kwargs.update(other.kwargs)
        return self

    def __call__(self, source: ABFLike) -> segment.Root:
        """
        When called with an abf file, construct a segment tree from its data and cache it.
        kwargs of the pipeline constructor are passed to the root of the tree
        :param source: ABF file or existing root
        :return: Root segment instance
        """
        if type(source) in ABFLikeTypes:
            source = as_abf(source)
            abfpath = Path(source.abfFilePath)
            key = abfpath.absolute()
            root = AbfRoot
        else:
            raise NanotraceException(f"{type(source)} is not a valid source type")

        if not key in self._cache:
            logger.debug("Creating tree from root: %s", key)
            rt = root(source, self.stages, pipeline=self, features=self.features, **self.kwargs)
            # Absolute file path is used as a key for caching, could use file hash
            self._cache[key] = rt
        logger.debug("Returning cached tree")
        return self._cache[abfpath]
