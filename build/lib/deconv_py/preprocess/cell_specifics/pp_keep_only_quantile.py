from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator


class PpKeepOnlyQuantile(TransformerMixin,BaseEstimator):
    def __init__(self,A_iBAQ,quantile=0.4):
        self.list_of_proteins=A_iBAQ[A_iBAQ>A_iBAQ.quantile(quantile)].dropna().index.tolist()

    def transform(self, data, *_):
        A, B = data[0], data[1]

        return A.loc[self.list_of_proteins],B.loc[self.list_of_proteins]


    def fit(self, *_):
        return self