from unittest import TestCase
from deconv_py.infras.global_utils import GlobalUtils
import pandas as pd
from deconv_py.infras.data_factory import DataFactory
from deconv_py.preprocess.intra_variance.aggregate_intra_variance import AggregateIntraVariance

class TestGlobalUtils(TestCase):
    def test_get_corospanding_columns_map(self):
        deconv = pd.read_pickle("C:\Repos\deconv_py\deconv_py\deconv")
        known = pd.read_pickle("C:\Repos\deconv_py\deconv_py\known")
        res = GlobalUtils.get_corospanding_mixtures_map(deconv, known)
        print(res)


class TestGlobalUtils(TestCase):
    def test_get_corospanding_cell_map(self):
        pp =  AggregateIntraVariance()
        data_f = DataFactory()

        source, _ = data_f.load_IBD_all_vs("Intensity", index_func=lambda x: x,
                                                          log2_transformation=True)
        target,_ = pp.transform([source,_])

        res = GlobalUtils.get_corospanding_cell_map(source, target)
        print(res)
        self.fail()
