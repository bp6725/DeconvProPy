from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import infras.cache_decorator as cache_decorator


class AggregateIntraVariance(TransformerMixin,BaseEstimator):
    def __init__(self,how = "median"):
        '''

        :param how:how to agg. mean ; med ; first : max ; majority
        '''
        self.how = how

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]
        full_profile_data = A.copy(deep=True).T

        full_profile_data["cell"] = full_profile_data.index.map(lambda x: x.split('_0')[0])

        if self.how == "mean" :
            res = full_profile_data.groupby("cell").mean().T

        if self.how == "median" :
            res = full_profile_data.groupby("cell").median().T

        if self.how == "max" :
            res = full_profile_data.groupby("cell").max().T

        if self.how == "first" :
            res = full_profile_data.groupby("cell").first().T
        try :
            res.deconv.transfer_all_relevant_properties(A)
            return [res, B]
        except :
            return [res, B]

    def fit(self, *_):
        return self