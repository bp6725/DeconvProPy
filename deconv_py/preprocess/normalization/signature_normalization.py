from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import infras.cache_decorator as cache_decorator


class SignatureNormalization(TransformerMixin,BaseEstimator) :
    def __init__(self,normalization_strategy = "max"):
        self.normalization_strategy = normalization_strategy


    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0].copy(deep=True), data[1].copy(deep=True)

        if self.normalization_strategy == "max" :
            norm_A = A
            norm_B = B.multiply(A.max().max()/B.max())

        if self.normalization_strategy == "mean" :
            norm_A = A
            norm_B = B.multiply(A.mean().mean()/B.mean())


        return norm_A,norm_B


    def fit(self, *_):
        return self
