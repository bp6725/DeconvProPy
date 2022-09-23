from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import infras.cache_decorator as cache_decorator

class PpEmpty(TransformerMixin,BaseEstimator):
    def __init__(self,keep_labels = False,with_label_prop=False):
        self.keep_labels = keep_labels

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]
        A_res = A
        if self.keep_labels :
            self._labels = {}
            self._labels["A"] = A
            self._labels["B"] = B

        try :
            A_res.deconv.transfer_all_relevant_properties(A)
            return [A_res, B]
        except :
            return [A_res, B]


    def fit(self, *_):
        return self