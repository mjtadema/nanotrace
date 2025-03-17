# Copyright 2025 Matthijs Tadema
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import pandas as pd
from anytree import NodeMixin, Resolver, LevelOrderGroupIter
from joblib import Parallel, delayed
from tqdm.asyncio import tqdm

from .decorators import requires_children
from .utils import PoolMixin, ReprMixin
import gc

logger = logging.getLogger(__name__)

name_resolver = Resolver('name')


class Segment(NodeMixin, PoolMixin, ReprMixin):
    """
    Segments make up the nodes and leaves of the tree.
    Segments have parent segments and child segments.
    """
    def __init__(
            self, t: np.ndarray, y: np.ndarray, l: list, stages: list, *,
            name=None, parent=None
    ):
        """
        :param t: array of time
        :param y: array of current
        :param l: list of additional labels passed upon segment initialization
        :param stages: list of pipeline stages
        :param nsegments: number of segments to generate, useful for testing. (default: -1 means generate all segments)
        :param name: str, name usually set to the value of __name__ of the refiner callable that generated the segment
        :param parent: parent Segment
        :param gc: garbage collect (default: False)
        """
        self.t = t
        self.y = y
        if len(l) > 0:
            self.l = l
        else:
            self.l = None
        self.parent = parent
        self.name = name
        self._root = None
        #TODO this could be simplified by simply taking parent.stages[1:] for each subsequent stage

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

    def __str__(self):
        return "Segment(%s) with %d datapoints" % (self.name, len(self.y))

    def __getitem__(self, item):
        """So we can use segments as "data" in plt.plot"""
        return getattr(self, item)

    # "Inherit" these properties from the root node
    @property
    def nsegments(self):
        return self.root.nsegments

    @property
    def gc(self):
        return self.root.gc

    @property
    def njobs(self):
        return self.root.njobs

    @NodeMixin.children.getter
    def children(self):
        """Lazily run self._refine if there are no children"""
        if not NodeMixin.children.fget(self):
            self.derive_children()
        return NodeMixin.children.fget(self)

    def derive_children(self):
        """
        Run the stage to derive children.
        Split in two possibilities: if the number of segments is specified we need to derive child segments in a loop.
        If not, we can use Parallel to derive children more efficiently.
        """
        if self.stage is not None:
            logger.debug("Segmenting with %s", self.stage.__name__)
            # if self.nsegments > 0:
            # logger.info(f"Only generating {self.nsegments}")
            for i, (t,y,*l) in enumerate(self.stage(self.t, self.y)):
                seg = Segment(t,y,l,stages=self.residual, name=self.stage.__name__)
                seg.parent = self
                if i == self.nsegments:
                    break

            # else:
            #     # Optimization: Generate new segments in parallel
            #     # Works best with generating many small segments
            #     for seg in Parallel(n_jobs=1)(delayed(Segment)(
            #             t,y,l, stages=self.residual, name=self.stage.__name__
            #     ) for t,y,*l in self.stage(self.t, self.y)):
            #         seg.parent = self
            #     if self.gc:
            #         # Unset the data arrays and run the garbage collector to save memory
            #         self.t = []
            #         self.y = []
            #         gc.collect()


class Root(NodeMixin, PoolMixin):
    """
    Special segment that acts as the interface to the pipeline, and the root of the tree of segments.
    As the main interface to the tree, Root implements some convenience functions and properties:
    """
    def __init__(self, stages, *, nsegments=-1, extractors=None, columns=None, pipe=None):
        """
        Root constructor takes an abf file and a pipeline of refi
        :param pipeline: a list of functions acting as pipeline stages
        :param extractors: a list of extractors to extract features from events
        :param columns: a list of column names to add to the features dataframe
        :param abf:
        :param gc: bool, garbage collect (default: False)
        :param njobs: number of jobs to run in parallel
        """
        self._features = None # Cache features

        self.pipe = pipe
        if extractors is None:
            extractors = []
        self.extractors = extractors
        self.columns = columns
        self.stages = stages
        self.nsegments = nsegments

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.by_index[item]
        elif isinstance(item, str):
            # Makes Root usable as a key addressable object
            if hasattr(self, item):
                return getattr(self, item)
            else:
                return self.by_name[item]
        else:
            raise TypeError("item must be either str or int, is %s", type(item))

    @property
    def njobs(self):
        return self.pipe.njobs

    @property
    def gc(self):
        return self.pipe.gc

    @property
    @requires_children
    def features(self):
        """
        Centrally extract features from events so that we can pool them and extract in parallel
        """
        #TODO this needs some refactoring
        # Cache features
        if self._features is None:
            # Optimization: calculate features for events in parallel ahead of time
            features = []
            cols = []
            for extractor in self.extractors:
                extracted = Parallel(n_jobs=self.njobs, backend='multiprocessing')(
                    delayed(extractor)(event.t,event.y)
                    for event in tqdm(self.events, desc="extracting features %s"%extractor.__name__)
                )
                extracted = np.array(extracted)
                if len(extracted.shape) == 1:
                    extracted = extracted[...,None]
                features.append(extracted)
                cols.extend([extractor.__name__+'_%d'%i for i in range(extracted.shape[-1])])
            if self.events[0].l is not None:
                cols.append("label")
                labels = []
                for event in self.events:
                    labels.append(event.l)
                features.append(labels)
            if not self.columns is None:
                cols = self.columns
            if len(features) > 0:
                self._features = pd.DataFrame(np.hstack(features), columns=cols)
        return self._features


    @property
    @requires_children
    def by_index(self):
        """
        returns a list of nodes grouped by level
        """
        return list(LevelOrderGroupIter(self))

    @property
    @requires_children
    def by_name(self):
        """
        returns a dict of nodes grouped by stage
        :return:
        """
        return {
            (stage.__name__ if callable(stage) else stage): level
            for stage, level in zip(
                ['root','sweep', *self.by_name],
                LevelOrderGroupIter(self)
            )
        }

    @property
    @requires_children
    def events(self):
        """Return segments from the lowest level"""
        return self.by_index[-1]

