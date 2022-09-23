from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import infras.cache_decorator as cache_decorator


class PpCleanIrrelevantProteins(TransformerMixin,BaseEstimator):
    def __init__(self):
        pass

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]
        _A, _B = A[A.columns[A.sum() > 0]], B

        _a_relevant_proteins = A[A.sum(axis=1) > 0].index
        _b_relevant_proteins = B[B.sum(axis=1) > 0].index

        _relevant_proteins = _a_relevant_proteins.intersection(_b_relevant_proteins)

        A_res = _A.loc[_relevant_proteins]

        try :
            A_res.deconv.transfer_all_relevant_properties(A)
            return [A_res, _B.loc[_relevant_proteins]]
        except :
            return [A_res, _B.loc[_relevant_proteins]]


    def fit(self, *_):
        return self