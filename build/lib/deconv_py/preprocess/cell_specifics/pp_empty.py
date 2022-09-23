from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator


class PpEmpty(TransformerMixin,BaseEstimator):
    def __init__(self,keep_labels = False,with_label_prop=False):
        self.keep_labels = keep_labels


    def transform(self, data, *_):
        A, B = data[0], data[1]
        if self.keep_labels :
            self._labels = {}
            self._labels["A"] = A
            self._labels["B"] = B
        return [A, B]


    def fit(self, *_):
        return self