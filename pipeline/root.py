import numpy as np
import pandas as pd
from anytree import NodeMixin, LevelOrderGroupIter
from joblib import Parallel, delayed
from pyabf import abfWriter
from tqdm.asyncio import tqdm

from .decorators import requires_children
from .segment import logger, Segment
from .utils import PoolMixin, ReprMixin


class Root(NodeMixin, PoolMixin, ReprMixin):
    """
    Special segment that acts as the head of the pipeline, and the root of the tree of segments
    """
    def __init__(self, abf, pipeline, extractors=None, columns=None, keep_steps=True):
        """
        Root constructor takes an abf file and a pipeline of refi
        :param pipeline: a list of functions acting as pipeline stages
        :param extractors: a list of extractors to extract features from events
        :param columns: a list of column names to add to the features dataframe
        :param abf:
        :param keep_steps: bool, whether to keep intermediate segments
        """
        self._features = None # Cache features
        self.abf = abf
        self.name = 'abf'
        self.pipeline = pipeline
        self.extractors = extractors
        self.columns = columns
        self.keep_steps = keep_steps

        logger.debug("Extracting %d sweeps", self.abf.sweepCount)
        self.sweeps = []
        for i in range(self.abf.sweepCount):
            self.abf.setSweep(i)
            self.sweeps.append(Segment(self.abf.sweepX, self.abf.sweepY, [], pipeline, name='sweep', parent=self, keep_steps=True))

    def __str__(self):
        return "Root from ABF (%s)" % (self.abf.abfFilePath)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.levels[item]
        elif isinstance(item, str):
            return self.stages[item]
        else:
            raise TypeError("item must be either str or int, is %s", type(item))

    @property
    def events(self):
        # Return segments from the lowest level
        return self.levels[-1]

    @property
    def fs(self):
        return self.abf.sampleRate

    @property
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
    @requires_children
    def levels(self):
        """
        returns a list of nodes grouped by level
        """
        return list(LevelOrderGroupIter(self))

    @property
    @requires_children
    def stages(self):
        """
        returns a dict of nodes grouped by stage
        :return:
        """
        return {
            (stage.__name__ if callable(stage) else stage): level
            for stage, level in zip(
                ['root','sweep',*self.pipeline],
                LevelOrderGroupIter(self)
            )
        }

    def to_abf(self, filename):
        """
        Write events to ABF v1 file
        :param filename:
        :return:
        """
        maxlen = max([len(event.y) for event in self.events])
        sweeps = []
        for event in self.events:
            padlen = maxlen - len(event.y)
            sweeps.append(np.pad(event.y, (0,padlen), mode='constant', constant_values=0))
        logger.debug("Writing %d sweeps to %s", len(sweeps), filename)
        abfWriter.writeABF1(np.asarray(sweeps), filename, sampleRateHz=self.fs)
