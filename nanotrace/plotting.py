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
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def dens2d(self, col1: str, col2: str, *args, **kwargs):
    """Wrapper around plot kind 'scatter' where the markers are colored by kde"""
    df = np.asarray(self._parent.loc[:,[col1,col2]])
    feat = np.asarray(df).T
    k = gaussian_kde(feat)
    Z = k(feat)
    norm = plt.Normalize(Z.min(), Z.max())
    return self._parent.plot(col1, col2, 'scatter', *args, color=norm(Z), **kwargs)

