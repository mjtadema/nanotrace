import logging

from anytree import NodeMixin
from joblib import Parallel, delayed

from .utils import PoolMixin, ReprMixin
import gc

logger = logging.getLogger(__name__)


class Segment(NodeMixin, PoolMixin, ReprMixin):
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
                    logger.debug("Found %d extractors", len(func))
                    self.extractors.extend(list(func))
                # Callables are treated as refiners
                elif callable(func):
                    logger.debug("Setting refiner: %s", func.__name__)
                    self.refiner = func
                    # Leave the rest of the pipeline as residual
                    self.residual = pipeline[i + 1:]
                    break
                else:
                    raise Exception("Unknown function type '%s': %s" % (type(func), func.__name__))

    @property
    def abf(self):
        return self.parent.abf

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
            logger.debug("Segmenting with %s", self.refiner.__name__)
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


