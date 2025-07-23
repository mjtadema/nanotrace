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
import warnings
from typing import Sequence, Callable

import numpy as np
import pandas as pd
from anytree import NodeMixin, Resolver, LevelOrderGroupIter, RenderTree
from joblib import Parallel, delayed, wrap_non_picklable_objects
from matplotlib import pyplot as plt
from pyabf import abfWriter
from tqdm.auto import tqdm
from ipywidgets import interact

from . import pipeline
from .exception import RootError, StageError

logger = logging.getLogger(__name__)

name_resolver = Resolver('name')

class Node(NodeMixin):
    """
    Generic tree node class with specific stuff for nanotrace.
    """
    def __init__(self, stages: list, *, name=None, parent=None):
        """
        Initialize a node

        :param stages: list of pipeline stages
        :param name: str, name usually set to the value of __name__ of the refiner callable that generated the segment
        :param parent: parent Segment
        """
        self.stages = stages
        self.name = name
        self.parent = parent
        self.idx = None

        # Consume part of the pipeline stages
        logger.debug(f"{stages=}")
        if stages:
            self.stage, *self.residual = stages
            if not callable(self.stage):
                raise TypeError("incompatible type '%s': %s" % (type(self.stage), self.stage.__name__))
        else:
            self.stage = None
            self.residual = []
        logger.debug(f"{self.stage=}, {self.residual=}")

    def __index__(self):
        return self.idx

    def __repr__(self):
        """Fancy tree rendering"""
        out = []
        render = iter(RenderTree(self))
        prev = None
        for pre, _, node in render:
            cnt = 0
            while prev == pre:
                # skip until we encounter next level
                pre, _, node = next(render)
                cnt += 1
            else:
                if cnt > 0:
                    out.append("%s ... Skipped %d segments" % (prev, cnt))
            out.append("%s%s" % (pre, str(node)))
            prev = pre
        return '\n'.join(out)

    @property
    def by_index(self) -> list[Segment]:
        """
        :return: a list of nodes grouped by level
        """
        return list(LevelOrderGroupIter(self))


class Root(Node):
    """
    Special node that acts as the interface to the datastructure, and the root of the tree of segments.
    As the main interface to the tree, Root implements some convenience functions and properties.
    Root takes care of calculating features over all events.
    """
    def __init__(self, *args, pipeline: pipeline.Pipeline, n_segments: int=-1,
                 features: Sequence | None=None, post: Callable | None=None, debug: bool=False, **kwargs) -> None:
        """
        Root constructor only sets up generic pipeline stuff.
        Specific roots that subclass this handle their specific tree setup.

        :param pipeline: an instance of the Pipeline that constructed this tree

        Optional parameters:
        :param n_segments: number of segments to generate. Useful for pipeline development.
        :param features: a list of extractors to extract features from events
        :param post: an optional list of postprocessors to apply to the events
        """
        super().__init__(*args, **kwargs)
        self._fs = None
        self._features = pd.DataFrame()

        self.pipe = pipeline
        if features is None:
            features = []
        self.extractors = features
        self.post = post
        self.debug = debug
        self.n_segments = n_segments

    @property
    def fs(self) -> int:
        return self._fs

    @property
    def n_jobs(self) -> int:
        return self.pipe.n_jobs

    @property
    def by_name(self) -> dict[str, tuple[Segment]]:
        """
        :return: a dict of tuples containing nodes grouped by stage
        """
        return {
            (stage.__name__ if callable(stage) else stage): level
            for stage, level in zip(
                ['root', 'sweep', *self.stages],
                LevelOrderGroupIter(self)
            )
        }

    @property
    def events(self) -> np.ndarray[Segment]:
        """
        :return: segments from the lowest level as array
        """
        events = np.asarray(self.by_index[-1])
        for i, event in enumerate(events):
            event.idx = i
        if self.post is None:
            return events
        else:
            return events[self.post(self.features)]

    def inspect(self, /, stage: str='sweep', ylim: None | tuple=None, **kwargs):
        """
        Interactive plot with a slider to view all segments and events from those segments
        at stage "stage"
        """
        parents = self.by_name[stage]
        @interact(i=(0, len(parents)-1, 1))
        def f(i=0):
            parents[i].inspect(**kwargs)
            if not ylim is None:
                plt.ylim(ylim)

    @property
    def features(self) -> pd.DataFrame:
        """
        Extract features from all events (lowest level of the tree)
        :return: dataframe of events of shape #events x #features
        """
        # Cache features
        if self._features.empty:
            # This used to use self.events, but that causes infinite recursion since that one uses these features now
            events = np.asarray(self.by_index[-1])
            # Doing it per extractor means we can parallelize efficiently
            # but requires some more bookkeeping to restructure the dataframe
            features = []
            columns = []
            # with tqdm(total=len(self.extractors)*len(self.events), desc="Calculating features") as progress:
            for extractor in self.extractors:
                # joblib.Parallel takes a generator
                extracted = Parallel(n_jobs=self.n_jobs)(
                    # That is called using joblib.delayed          with these arguments
                    delayed(wrap_non_picklable_objects(extractor))(event.t, event.y)
                    # For each event in this iterable
                    for event in tqdm(events, desc=f"{extractor.__name__}")           # by using tqdm in the generator we can monitor progress
                        #desc="extracting features %s" % extractor.__name__)
                )
                extracted = np.asarray(extracted)
                if len(extracted.shape) > 1:
                    columns.extend([extractor.__name__ + '_%d' % i for i in range(extracted.shape[-1])])
                    features.extend([*extracted.T])
                else:
                    columns.append(extractor.__name__)
                    features.append(extracted)
                # progress.update(len(self.events))


            # if events have a label (event.l), add it to features
            event: Segment = events[0] # This is an array of Segments
            if event.l is not None:
                columns.append("label")
                labels = []
                for event in events:
                    labels.append(event.l)
                features.append(labels)

            # Create the dataframe, if there are no features return an empty dataframe
            if len(features) > 0:
                self._features = pd.DataFrame(np.vstack(features).T, columns=columns)

        # Return cached features
        return self._features

    def write_abf(self, filename: str, *, fs: int=None) -> None:
        """
        Write events as sweeps to an ABF v1 file
        :param filename: filename to write to
        :param fs: sample rate in Hz
        :return:
        """
        if not fs:
            try:
                fs = self.fs
            except AttributeError:
                raise RootError("write_abf requires a sample rate")
        maxlen = max([len(event.y) for event in self.events])
        sweeps = []
        for event in self.events:
            padlen = maxlen - len(event.y)
            sweeps.append(np.pad(event.y, (0, padlen), mode='constant', constant_values=0))
        logger.debug("Writing %d sweeps to %s", len(sweeps), filename)
        abfWriter.writeABF1(np.asarray(sweeps), filename, sampleRateHz=fs)


class Segment(Node):
    """
    Segments make up the nodes and leaves of the tree.
    Segments have parent segments and child segments.
    Segments take care of generating events.
    """
    def __init__(self, t: np.ndarray, y: np.ndarray, l: list, *args, **kwargs):
        """
        :param t: array of time
        :param y: array of current
        :param l: list of additional labels passed upon segment initialization
        """
        self.t = t
        self.y = y
        if len(l) == 1:
            self.l = l[0]
        elif len(l) > 1:
            raise NotImplementedError("more than 1 label is not supported for now")
        else:
            self.l = None
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "Segment(%s) with %d datapoints" % (self.name, len(self.y))

    def __getitem__(self, item):
        """So we can use segments as "data" in plt.plot"""
        return getattr(self, item)

    def _run_stage(self):
        """
        Run the stage to derive children.
        """
        if self.stage is not None:
            # logger.debug("Segmenting with %s", self.stage.__name__)
            if self.n_segments > 0:
                logger.info(f"Generating {self.n_segments}")
            try:
                for i, (t, y, *l) in enumerate(self.stage(self.t, self.y)):
                    if len(l) == 0 and self.l is not None:
                        l = (self.l,) # if we already have a label, add it to the next stage
                    seg = Segment(t, y, l, stages=self.residual, name=self.stage.__name__)
                    seg.parent = self
                    if i == self.n_segments:
                        break
            except StageError as e:
                if self.root.debug:
                    raise e
                else:
                    warnings.warn("Stage %s failed"%self.stage.__name__)

    @Node.children.getter
    def children(self):
        """Lazily run self._run_stage if there are no children"""
        if not Node.children.fget(self):
            self._run_stage()
        return Node.children.fget(self)

    # "Inherit" these properties from the root node
    @property
    def n_jobs(self):
        return self.root.n_jobs

    @property
    def n_segments(self):
        return self.root.n_segments

    @property
    def events(self) -> np.ndarray:
        """
        :return: segments from the lowest level as array
        """
        events = np.asarray(self.by_index[-1])
        return events

    # Convenience functions
    def plot(self, fmt='', normalize=False, **kwargs):
        """Plot the time vs current of this segment"""
        if normalize:
            x = np.linspace(0,1, len(self.t))
        else:
            x = self.t
        y = self.y

        plt.plot(x, y, fmt, data=self, **kwargs)

    def inspect(self, *args, **kwargs):
        """Plot events on top of self"""
        self.plot(*args, **kwargs)
        for event in self.events:
            # if event has label, use it for coloring
            if event.l is not None:
                label = event.l
                kwargs['color'] = f"C{label}"
            event.plot(*args, **kwargs)

    def write_abf(self, filename: str, *, fs=None) -> None:
        """
        Write events as sweeps to an ABF v1 file
        :param filename: filename to write to
        :return:
        """
        if not fs:
            try:
                fs = self.fs
            except AttributeError:
                raise RootError("write_abf requires a sample rate")
        logger.debug("Writing %d datapoints to %s", len(self.y), filename)
        abfWriter.writeABF1(self.y, filename, sampleRateHz=fs)
