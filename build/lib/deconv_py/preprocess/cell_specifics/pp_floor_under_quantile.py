from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from deconv_py.infras.global_utils import GlobalUtils

class PpFloorUnderQuantile(TransformerMixin,BaseEstimator):

    def __init__(self,quantile = 0.1):
        self.quantile = quantile


    def transform(self, data, *_):
        A, B = data[0], data[1]

        A_res = A.copy(deep=True)
        _melted = A_res.melt()["value"]
        _quantile = _melted[_melted > 0].quantile(self.quantile)
        A_res[A_res  < _quantile] = 0

        try :
            A_res.deconv.transfer_all_relevant_properties(A)
            return [A_res,B]
        except :
            return [A_res,B]
