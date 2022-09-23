from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from deconv_py.infras.data_factory import DataFactory


class PickDataSet(TransformerMixin,BaseEstimator):
    def __init__(self,data_set_version = "original_data_all_vs"):
        self.data_set_version =  data_set_version

    def transform(self, data, *_):

        # if self.data_set_version == "original_data_all_vs" :
        #     data_factory = DataFactory()
        #     _A, _B = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
        #                                                      log2_transformation=True)
        #
        # if self.data_set_version == "original_data_not_imputed" :
        #     _A, _B = data_factory.load_no_imputation_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
        #                                                                          log2_transformation=False)

        _A, _B = data[self.data_set_version]

        return [_A,_B]


    def fit(self, *_):
        return self