import logging
from pathlib import Path

from .root import Root
from .utils import ABFLike, as_abf

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline factory with caching
    """
    def __init__(self, *pipeline, **kwargs):
        """
        Pipeline constructor takes a list of functions that make up the pipeline that we refer to as "refiners".
        These can be any callable, but they must take two arrays as arguments (time and current arrays)
        and return an _iterable_ of time and current arrays.
        Typically these are generators for simplicity.
        Refiners that don't return anything can be used to filter out unwanted segments
        Refiners that return an iterable with only one time and current array can be used to filter the data itself
        See
        :param pipeline:
        :param kwargs:
        """
        self._cache = {}
        logger.debug("Constructing pipeline with %d steps: %s", len(pipeline), ",".join([f.__name__ for f in pipeline]))
        self.pipeline = pipeline
        self.kwargs = kwargs

    def __str__(self):
        return "Pipeline: %s with %d stages" % (self.__name__, len(self.pipeline))

    def __call__(self, abf: ABFLike):
        """
        When called with an abf file, construct a segment tree from its data and cache it.
        kwargs of the pipeline constructor are passed to the root of the tree
        :param abf:
        :return: Root
        """
        abf = as_abf(abf)
        abfpath = Path(abf.abfFilePath)
        if not abfpath.absolute() in self._cache:
            logger.debug("Creating tree from %s", abfpath)
            # Absolute file path is used as a key for caching, could use file hash
            self._cache[abfpath] = Root(abf, self.pipeline, **self.kwargs)
        logger.debug("Returning cached tree")
        return self._cache[abfpath]
