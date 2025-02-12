from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from anytree import PreOrderIter, LevelOrderGroupIter, NodeMixin
from joblib import Parallel, delayed
from pyabf import ABF
from tqdm.auto import tqdm

from .utils import PoolMixin
from .segment import Segment

################
### PIPELINE ###
################

class AbfRoot(NodeMixin, PoolMixin):
    """
    Special segment that acts as the head of the pipeline
    """
    def __init__(self, pipeline, extractors=None, columns=None, abf=None, keep_steps=True):
        self.abf = abf
        self.t = []
        self.y = []
        self.sweeps = []
        self.name = 'abf'
        self._features = None
        self.extractors = extractors
        self.columns = columns
        self.keep_steps = keep_steps
        for i in range(self.abf.sweepCount):
            self.abf.setSweep(i)
            self.sweeps.append(Segment(self.abf.sweepX, self.abf.sweepY, [], pipeline, name='sweep', parent=self, keep_steps=True))

    def __str__(self):
        return "ABF from %s" % (self.abf.abfFilePath)

    @property
    def events(self):
        return self.levels[-1]

    @property
    def features(self):
        """
        Pipeline root only pools features from sweeps
        """
        # Cache features
        if self.extractors is None: return
        if self._features is None:
            # Optimization: calculate features for events in parallel ahead of time
            features = []
            cols = []
            for extractor in self.extractors:
                extracted = Parallel(n_jobs=4, backend='multiprocessing')(
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
    def levels(self):
        """
        returns a list of nodes grouped by level
        conceptually as "stages" of the pipeline"
        """
        if not self.children:
            # Generate the tree
            for node in PreOrderIter(self):
                pass
        return list(LevelOrderGroupIter(self))

class Pipeline:
    """Pipeline factory with caching"""
    def __init__(self, *pipeline, **kwargs):
        self.name = 'root'
        self.pipeline = pipeline
        self._cache = {}
        self.kwargs = kwargs

    def __call__(self, abfpath: Path):
        abfpath = Path(abfpath).absolute()
        # Could even use the file hash perhaps...
        if not abfpath in self._cache:
            abf = ABF(abfpath)
            self._cache[abfpath] = AbfRoot(self.pipeline, abf=abf, **self.kwargs)
        return self._cache[abfpath]

