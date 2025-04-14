__copyright__ = """
Copyright 2025 Matthijs Tadema

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

import numpy as np
from pyabf import abfWriter

from .segment import Segment
from root import Root
from .utils import ReprMixin

logger = logging.getLogger(__name__)


class AbfRoot(Root, ReprMixin):
    """
    Root node for abf files
    """

    def __init__(self, abf, *args, **kwargs) -> None:
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
            if 0 < self.n_segments <= i:
                break

    def __str__(self) -> str:
        return "Root from %s (%s)" % (self.name, self.abf.abfFilePath)

    @property
    def fs(self) -> int:
        return self.abf.sampleRate

    def to_abf(self, filename: str) -> None:
        """
        Write events as sweeps to an ABF v1 file
        :param filename: filename to write to
        :return:
        """
        maxlen = max([len(event.y) for event in self.events])
        sweeps = []
        for event in self.events:
            padlen = maxlen - len(event.y)
            sweeps.append(np.pad(event.y, (0, padlen), mode='constant', constant_values=0))
        logger.debug("Writing %d sweeps to %s", len(sweeps), filename)
        abfWriter.writeABF1(np.asarray(sweeps), filename, sampleRateHz=self.fs)
