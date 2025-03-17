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
        if self.extractors is None: return
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
            if not self.columns is None:
                cols = self.columns
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

