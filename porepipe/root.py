from typing import Any

import numpy as np
import pandas as pd
from anytree import NodeMixin, LevelOrderGroupIter
from joblib import Parallel, delayed, wrap_non_picklable_objects
from tqdm.asyncio import tqdm

from decorators import requires_children
from segment import logger, Segment
from utils import PoolMixin


class Root(NodeMixin, PoolMixin):
    """
    Special segment that acts as the interface to the pipeline, and the root of the tree of segments.
    As the main interface to the tree, Root implements some convenience functions and properties:
    """

    def __init__(self, stages, *, n_segments=-1, features=None, post=None, columns=None, pipe=None) -> None:
        """
        Root constructor takes an abf file and a pipeline of refi
        :param pipeline: a list of functions acting as pipeline stages
        :param extractors: a list of extractors to extract features from events
        :param columns: a list of column names to add to the features dataframe
        :param abf:
        :param gc: bool, garbage collect (default: False)
        :param njobs: number of jobs to run in parallel
        """
        self._features = None  # Cache features

        self.pipe = pipe
        if features is None:
            features = []
        self.extractors = features
        self.columns = columns
        self.stages = stages
        self.post = post
        self.n_segments = n_segments

    def __getitem__(self, item) -> Any:
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
    def n_jobs(self) -> int:
        return self.pipe.n_jobs

    @property
    def gc(self) -> bool:
        return self.pipe.gc

    @property
    @requires_children
    def features(self) -> pd.DataFrame:
        """
        Centrally extract features from events so that we can pool them and extract in parallel
        """
        # TODO this needs some refactoring
        # Cache features
        if self._features is None:
            # Optimization: calculate features for events in parallel in one go
            features = []
            cols = []
            for extractor in self.extractors:
                extracted = Parallel(n_jobs=self.n_jobs)(
                    delayed(wrap_non_picklable_objects(extractor))(event.t, event.y)
                    for event in tqdm(np.asarray(self.by_index[-1]), desc="extracting features %s" % extractor.__name__)
                )
                logger.debug(f"{len(extracted)=}")
                extracted = np.array(extracted)
                if len(extracted.shape) == 1:
                    extracted = extracted[..., None]
                features.append(extracted)
                cols.extend([extractor.__name__ + '_%d' % i for i in range(extracted.shape[-1])])
            # This used to use self.events, but that causes infinite recursion since that one uses these features now
            if np.asarray(self.by_index[-1])[0].l is not None:
                cols.append("label")
                labels = []
                for event in np.asarray(self.by_index[-1]):
                    labels.append(event.l)
                features.append(labels)
            if not self.columns is None:
                cols = self.columns
            if len(features) > 0:
                self._features = pd.DataFrame(np.hstack(features), columns=cols)
        return self._features

    @property
    @requires_children
    def by_index(self) -> list[Any]:
        """
        returns a list of nodes grouped by level
        """
        return list(LevelOrderGroupIter(self))

    @property
    @requires_children
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
    @requires_children
    def events(self) -> np.ndarray:
        """Return segments from the lowest level"""
        if not self.post is None:
            return np.asarray(self.by_index[-1])[self.post(self.features)]
        return np.asarray(self.by_index[-1])
