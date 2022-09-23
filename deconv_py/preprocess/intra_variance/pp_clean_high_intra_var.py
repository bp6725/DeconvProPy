from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from infras.deconv_data_frame import DecovAccessor
import infras.cache_decorator as cache_decorator

class PpCleanHighIntraVar(TransformerMixin,BaseEstimator):
    def __init__(self,how = "std",std_trh = 2,range_trh = 2,number_of_zeros = 2):
        """
        how : filter method. "std" : normalize std. "range" : (max-min)/mean. "zeros" : number of zeros
        std_trh : filter out biggest intra std (normalize)
        """
        self.how = how
        self.std_trh = std_trh
        self.range_trh = range_trh
        self.number_of_zeros = number_of_zeros

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]
        full_profile_data = A.copy(deep=True).T
        full_profile_data["cell_for_gb"] = full_profile_data .index.map(lambda x: x.split('_0')[0])

        if self.how == "std" :
            full_profile_data = (full_profile_data.groupby("cell_for_gb").std() / (
                        full_profile_data.groupby("cell_for_gb").mean() + 0.001)).T
            full_profile_data = full_profile_data[full_profile_data < self.std_trh].dropna(how="all")

        if self.how == "range" :
            full_profile_data = ((full_profile_data.groupby("cell_for_gb").max() - full_profile_data.groupby("cell_for_gb").min()) / (
                    full_profile_data.groupby("cell_for_gb").mean() + 0.001)).T
            full_profile_data = full_profile_data[full_profile_data < self.range_trh].dropna(how="all")

        if self.how == "number_of_zeros" :
            full_profile_data = full_profile_data.groupby("cell_for_gb").agg(lambda x: x.eq(0).sum())
            full_profile_data = full_profile_data[full_profile_data < self.number_of_zeros].dropna(how="all")

        _A,_B = A.reindex(full_profile_data.index),B
        _A.deconv.transfer_all_relevant_properties(A)

        _A.deconv.update_intra_variance_dict({self.how:full_profile_data})
        _A.deconv.set_intra_variance_trh({"std" :self.std_trh ,"range" : self.range_trh,"number_of_zeros" : self.number_of_zeros})
        _A.deconv.set_agg_cells(A.deconv.is_agg_cells_profile)

        try :
            return [_A,_B]
        except :
            return [_A,_B]


    def fit(self, *_):
        return self