from pathlib import Path

from anytree import PreOrderIter, LevelOrderGroupIter, NodeMixin
from pyabf import ABF

from .utils import PoolMixin
from .segment import Segment


################
### PIPELINE ###
################

class AbfRoot(NodeMixin, PoolMixin):
    """
    Special segment that acts as the head of the pipeline
    """
    def __init__(self, pipeline, abf=None):
        self.abf = abf
        self.t = []
        self.y = []
        self.sweeps = []
        self.name = 'abf'
        self._features = []
        for i in range(self.abf.sweepCount):
            self.abf.setSweep(i)
            self.sweeps.append(Segment(self.abf.sweepX, self.abf.sweepY, pipeline, name='sweep', parent=self))
        # Generate the tree
        # TODO remove this but make sure the tree gets generated when we get features etc
        for node in PreOrderIter(self):
            pass

    def __str__(self):
        return "ABF from %s" % (self.abf.abfFilePath)

    @property
    def events(self):
        return self.leaves

    @property
    def features(self):
        """
        Pipeline root only pools features from sweeps
        """
        # Cache features
        if self._features is None:
            self._features = self.pool()
        return self._features

    @property
    def stages(self):
        """
        returns a list of nodes grouped by level
        conceptually as "stages" of the pipeline"
        """
        return list(LevelOrderGroupIter(self))

class Pipeline:
    def __init__(self, *pipeline):
        self.name = 'root'
        self._features = None
        self.pipeline = pipeline
        self._cache = {}

    def __call__(self, abfpath: Path):
        abfpath = Path(abfpath).absolute()
        # Could even use the file hash perhaps...
        if not abfpath in self._cache:
            abf = ABF(abfpath)
            self._cache[abfpath] = AbfRoot(self.pipeline, abf=abf)
        return self._cache[abfpath]

