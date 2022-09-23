from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import infras.cache_decorator as cache_decorator


class PpKeepOnlyQuantile(TransformerMixin,BaseEstimator):
    def __init__(self,A_iBAQ,quantile=0.4):
        self.list_of_proteins=A_iBAQ[A_iBAQ>A_iBAQ.quantile(quantile)].dropna().index.tolist()

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]

        list_of_proteins = self.list_of_proteins

        if A.deconv.must_contain_genes is not None :
            list_of_proteins = list(set(list_of_proteins + A.deconv.must_contain_genes.to_list()))

        return A.loc[list_of_proteins],B.loc[list_of_proteins]


    def fit(self, *_):
        return self