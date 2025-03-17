import logging

import numpy as np
from pyabf import abfWriter

from .segment import Root, Segment
from .utils import ReprMixin

logger = logging.getLogger(__name__)


class AbfRoot(Root, ReprMixin):
    """
    Root node for abf files
    """
    def __init__(self, abf, *args, **kwargs):
        self.name = 'abf'
        self.abf = abf
        super().__init__(*args, **kwargs)

        logger.debug("Segmenting abf file with %d sweeps", self.abf.sweepCount)
        self.sweeps = []
        for i in range(self.abf.sweepCount):
            self.abf.setSweep(i)
            self.sweeps.append(
                Segment(
                    self.abf.sweepX, self.abf.sweepY, [], self.stages,
                    name='sweep', parent=self
                )
            )
            if 0 < self.nsegments <= i:
                break

    def __str__(self):
        return "Root from %s (%s)" % (self.name, self.abf.abfFilePath)

    @property
    def fs(self):
        return self.abf.sampleRate

    def to_abf(self, filename: str):
        """
        Write events as sweeps to an ABF v1 file
        :param filename: filename to write to
        :return:
        """
        maxlen = max([len(event.y) for event in self.events])
        sweeps = []
        for event in self.events:
            padlen = maxlen - len(event.y)
            sweeps.append(np.pad(event.y, (0,padlen), mode='constant', constant_values=0))
        logger.debug("Writing %d sweeps to %s", len(sweeps), filename)
        abfWriter.writeABF1(np.asarray(sweeps), filename, sampleRateHz=self.fs)
