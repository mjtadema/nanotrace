from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
Segment = type('Segment', (object,), {}) # for type hinting

def segment(segment, fmt='', **kwargs):
    plt.plot('t','y',fmt,data=segment,**kwargs)


def events(segment, fmt='', color_by=None, zoom=False, **kwargs):
    for event in segment.events:
        if color_by == 'label':
            # event.l is a list of values
            # 'label' implies that it has a single integer value
            kwargs['color'] = f"C{int(event.l[0])}"
        plt.plot('t','y',fmt,data=event, **kwargs)
    if zoom:
        cat_y = np.concatenate([event.y for event in segment.events])
        plt.ylim((cat_y.min(), cat_y.max()))
        cat_t = np.concatenate([event.t for event in segment.events])
        plt.xlim((cat_t.min(), cat_t.max()))


class PlotMixin:
    def plot(self: Segment, **kwargs):
        return segment(self, **kwargs)
    def plot_events(self: Segment, **kwargs):
        return events(self, **kwargs)