from anytree import NodeMixin, RenderTree
import pandas as pd

from .utils import PoolMixin


class Segment(NodeMixin, PoolMixin):
    def __init__(self, t, y, pipeline, name=None, parent=None, label=None):
        self.t = t
        self.y = y
        self.parent = parent
        self.abf = self.parent.abf
        self.name = name
        self.refiner = None
        self.extractors = []
        self.condensors = []
        self.residual = []
        self.label = label  # Extra label given by refiner

        # Consume part of the pipeline
        pipeline = list(pipeline)
        if len(pipeline) > 0:
            for i, func in enumerate(pipeline):
                # Take extractors until a refiner is encountered
                if func.tag == 'extractor':
                    assert len(self.condensors) == 0
                    self.extractors.append(func)
                elif func.tag == 'condensor':
                    assert len(self.extractors) == 0
                    self.condensors.append(func)
                elif func.tag == 'refiner':
                    self.refiner = func
                    break
            # Leave the rest of the pipeline as residual
            self.residual = pipeline[i + 1:]

    def __repr__(self):
        """Fancy tree rendering"""
        out = []
        render = iter(RenderTree(self))
        prev = None
        for pre, _, node in render:
            cnt = 0
            while prev == pre:
                # skip until we encounter next level
                pre, _, node = next(render)
                cnt += 1
            else:
                if cnt > 0:
                    out.append("%s ... Skipped %d segments" % (prev, cnt))
            out.append("%s%s" % (pre, str(node)))
            prev = pre
        return '\n'.join(out)

    def __str__(self):
        return "Segment(%s) with %d datapoints" % (self.name, len(self.y))

    def __getitem__(self, item):
        """So we can use segments as "data" in plt.plot"""
        return getattr(self, item)

    @NodeMixin.children.getter
    def children(self):
        """Automatically run self._refine if there are no children"""
        if not NodeMixin.children.fget(self):
            self.refine()
        return NodeMixin.children.fget(self)

    def refine(self):
        """Wrap self.refine to generate new segments"""
        if self.refiner is not None:
            for t, y, *l in self.refiner(self.t, self.y):
                l = l[0] if len(l) > 0 else None
                Segment(t, y, pipeline=self.residual, name=self.refiner.__name__, parent=self, label=l)

    def extract(self):
        """
        Extract new features from self
        """
        # If there are any additional features specified already add them to the df
        cols = []
        features = []
        for extract in self.extractors:
            cols.append(extract.__name__)
            features.append(extract(self.t, self.y))
        features = pd.DataFrame([features], columns=cols)
        return features

    # def pool(self):
    #     """
    #     Pool features from children
    #     """
    #     if self.children:
    #         return pd.concat([child.features for child in self.children], ignore_index=True)
    #     else:
    #         return pd.DataFrame()

    def condense(self):
        """
        If there are no condensors, simply pool features from children
        """
        pooled = self.pool()
        if len(self.condensors) > 0:
            cols = []
            features = []
            for condense in self.condensors:
                cols.append(condense.__name__)
                features.append(condense(pooled))
            features = pd.DataFrame([features], columns=cols)
            return features
        else:
            return pooled

    @property
    def features(self):
        """
        Wrapper around extractors to extract features.
        If no extractors are defined, condense features from children
        """
        condensed = self.condense()
        extracted = self.extract()
        features = pd.concat([extracted, condensed], axis=1)
        # Add extra label if we have one
        if self.label is not None:
            features[self.name] = self.label
        return features.dropna()

