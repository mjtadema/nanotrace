"""
Functions, classes and types to handle abf files.
"""
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
from pathlib import Path
from typing import Union

from pyabf import ABF

from .segment import Segment, Root

logger = logging.getLogger(__name__)


ABFLike = Union[ABF, str, Path]
ABFLikeTypes = [ABF, str, Path]


def as_abf(abf: ABFLike) -> ABF:
    if isinstance(abf, str):
        abf = Path(abf)
    if isinstance(abf, Path):
        if not abf.exists(): raise FileNotFoundError(abf)
        abf = ABF(abf)
    if not type(abf) in ABFLikeTypes:
        raise TypeError(('Expected an AbfLike, not type', type(abf)))
    return abf


class AbfRoot(Root):
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


