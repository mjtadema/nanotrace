import random

from anytree import NodeMixin, RenderTree
import pandas as pd
from joblib import Parallel, delayed
from .utils import PoolMixin
from tqdm.auto import tqdm
import gc


class Segment(NodeMixin, PoolMixin):
    def __init__(self, t, y, l, pipeline, name=None, parent=None, keep_steps=True):
        self.t = t
        self.y = y
        self.l = l # Extra label given by refiner
        self.parent = parent
        self.name = name
        self.refiner = None
        self.keep_steps = keep_steps
        self.extractors = []
        self.condensors = []
        self.residual = []
        self._features = None

        # Consume part of the pipeline
        pipeline = list(pipeline)
        if len(pipeline) > 0:
            for i, func in enumerate(pipeline):
                # if func is actually a tuple, treat as extractors
                if isinstance(func, tuple):
                    self.extractors.extend(list(func))
                # Callables are treated as refiners
                elif hasattr(func, '__call__'):
                    self.refiner = func
                    # Leave the rest of the pipeline as residual
                    self.residual = pipeline[i + 1:]
                    break

    @property
    def abf(self):
        return self.parent.abf

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

    def __str__(self):
        return "Segment(%s) with %d datapoints" % (self.name, len(self.y))

    def __getitem__(self, item):
        """So we can use segments as "data" in plt.plot"""
        return getattr(self, item)

    @NodeMixin.children.getter
    def children(self):
        """Automatically run self._refine if there are no children"""
        if not NodeMixin.children.fget(self):
            self.refine()
        return NodeMixin.children.fget(self)

    def refine(self):
        """Wrap self.refine to generate new segments"""
        if self.refiner is not None:
            # Optimization: Generate new segments in parallel
            # Works best with generating many small segments
            for seg in Parallel(n_jobs=8)(delayed(Segment)(
                    t,y,l,pipeline=self.residual, name=self.refiner.__name__
            ) for t,y,*l in self.refiner(self.t, self.y)):
                seg.parent = self
            if not self.keep_steps:
                self.t = []
                self.y = []
                gc.collect()

    # def extract(self):
    #     """
    #     Extract new features from self
    #     """
    #     # If there are any additional features specified already add them to the df
    #     return #run_extractors(self.t, self.y, *self.extractors)

    # def pool(self):
    #     """
    #     Pool features from children
    #     """
    #     if self.children:
    #         return pd.concat([child.features for child in self.children], ignore_index=True)
    #     else:
    #         return pd.DataFrame()

    # def condense(self):
    #     """
    #     If there are no condensors, simply pool features from children
    #     """
    #     pooled = self.pool()
    #     if len(self.condensors) > 0:
    #         cols = []
    #         features = []
    #         for condense in self.condensors:
    #             cols.append(condense.__name__)
    #             features.append(condense(pooled))
    #         features = pd.DataFrame([features], columns=cols)
    #         return features
    #     else:
    #         return pooled

    # @property
    # def features(self):
    #     """
    #     Wrapper around extractors to extract features.
    #     If no extractors are defined, condense features from children
    #     """
    #     if self._features is None:
    #         raise NotImplementedError("Features are not yet extracted")
    #         # condensed = self.condense()
    #         extracted = self.extract()
    #         features = pd.concat([extracted, condensed], axis=1)
    #         # Add extra label if we have one
    #         if len(self.l) > 0:
    #             for i, label in enumerate(self.l):
    #                 features[self.name+"_%d"%i] = label
    #         self._features = features.dropna()
    #     return self._features
    #
    # @features.setter
    # def features(self, features):
    #     self._features = features
