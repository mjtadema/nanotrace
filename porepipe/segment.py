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
from typing import Any

import numpy as np
from anytree import NodeMixin, Resolver, LevelOrderGroupIter
from matplotlib import pyplot as plt

from .decorators import requires_children
from .utils import PoolMixin, ReprMixin

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
        # TODO this could be simplified by simply taking parent.stages[1:] for each subsequent stage

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
    def n_segments(self):
        return self.root.n_segments

    @property
    def gc(self):
        return self.root.gc

    @property
    def n_jobs(self):
        return self.root.n_jobs

    @NodeMixin.children.getter
    def children(self):
        """Lazily run self._refine if there are no children"""
        if not NodeMixin.children.fget(self):
            self.derive_children()
        return NodeMixin.children.fget(self)
    # TODO property for events

    @property
    @requires_children
    def by_index(self) -> list[Any]:
        """
        returns a list of nodes grouped by level
        """
        return list(LevelOrderGroupIter(self))

    @property
    def events(self) -> np.ndarray:
        """Return segments from the lowest level"""
        return np.asarray(self.by_index[-1])

    def derive_children(self):
        """
        Run the stage to derive children.
        Split in two possibilities: if the number of segments is specified we need to derive child segments in a loop.
        If not, we can use Parallel to derive children more efficiently.
        """
        if self.stage is not None:
            # logger.debug("Segmenting with %s", self.stage.__name__)
            # if self.nsegments > 0:
            logger.info(f"Only generating {self.n_segments}")
            for i, (t, y, *l) in enumerate(self.stage(self.t, self.y)):
                seg = Segment(t, y, l, stages=self.residual, name=self.stage.__name__)
                seg.parent = self
                if i == self.n_segments:
                    break
            #
            # else:
            #TODO  must swap joblib backend with "multiprocess" as it can serialize decorated functions
            #     # Optimization: Generate new segments in parallel
            #     # Works best with generating many small segments
            #     for seg in Parallel(n_jobs=self.root.n_jobs)(delayed(Segment)(
            #             t,y,l, stages=self.residual, name=self.stage.__name__
            #     ) for t,y,*l in self.stage(self.t, self.y)):
            #         seg.parent = self
            #     if self.gc:
            #         # Unset the data arrays and run the garbage collector to save memory
            #         self.t = []
            #         self.y = []
            #         gc.collect()

    def plot(self, fmt='', no_time=False, **kwargs):
        """Plot the time vs current of this segment"""
        if no_time:
            x = np.linspace(0,1, len(self.t))
        else:
            x = self.t
        y = self.y

        plt.plot(x, y, fmt, data=self,**kwargs)


