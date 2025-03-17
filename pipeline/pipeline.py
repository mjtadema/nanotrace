import logging
from pathlib import Path

from .abf import AbfRoot
from .utils import ABFLike, as_abf

logger = logging.getLogger(__name__)


class Pipeline:
    """
    The Pipeline class is the main class of this module.
    Its job is to define the pipeline through _stages_, functions that each modify timeseries data
    as steps in a in a pipeline.

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
    def __init__(self, *stages, **kwargs):
        """
        A pipeline is constructed as a linear list of pipeline "stages".

        :param stages: a list of stages (callables) that make up the pipeline steps
        :param kwargs: additional keyword arguments are passed to the root segment
        """
        # The pipeline instance caches the root segment with the abf file paths as keys
        self._cache = {}
        logger.debug("Constructing pipeline with %d steps: %s", len(stages), ",".join([f.__name__ for f in stages]))
        self.stages = stages
        self.kwargs = kwargs

    def __str__(self):
        repr = "Pipeline with %d stage(s): " % (len(self.stages))
        repr += ', '.join([stage.__name__ for stage in self.stages])
        return repr

    def __repr__(self):
        return str(self)

    def __call__(self, abf: ABFLike, njobs=1, gc=False, cache=True):
        """
        When called with an abf file, construct a segment tree from its data and cache it.
        kwargs of the pipeline constructor are passed to the root of the tree
        :param abf: ABF file
        :return: Root segment instance
        """
        #TODO this is a bit messy
        self.njobs = njobs
        self.gc = gc
        abf = as_abf(abf)
        abfpath = Path(abf.abfFilePath)
        if not abfpath.absolute() in self._cache:
            logger.debug("Creating tree from %s", abfpath)
            rt = AbfRoot(abf, self.stages, pipe=self, **self.kwargs)
            if not cache:
                # Don't cache if testing
                return rt
            # Absolute file path is used as a key for caching, could use file hash
            self._cache[abfpath] = rt
        logger.debug("Returning cached tree")
        return self._cache[abfpath]
