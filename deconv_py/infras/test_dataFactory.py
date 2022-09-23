from unittest import TestCase
from deconv_py.infras.data_factory import DataFactory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


class TestDataFactory(TestCase):
    def test_load_simple_artificial_profile(self):
        data_f = DataFactory()
        res = data_f.load_simple_artificial_profile(intensity_type="Intensity", index_func=lambda x: x.split(";")[0],
                                                    sample_to_pick="all")
        print("finish")
        self.fail()

    def test_load_simple_IBD_profile(self):
        self.fail()

    def test_load_IBD_all_vs(self):
        data_f = DataFactory()

        A_intensity, B_intensity = data_f.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True,auto_filter_by=True)
        # A_iBAQ, B_iBAQ = data_f.load_IBD_all_vs("iBAQ", index_func=lambda x: x.split(";")[0], log2_transformation=True)

        print("finish")
        self.assertTrue(True)

    def test_load_IBD_vs_A_and_B_intensity(self):
        self.fail()

    def test_build_simulated_data(self):
        data_f = DataFactory()
        _, X, B = data_f.build_simulated_data()
        self.fail()

    def test__clac_chance_to_be_zero(self):
        data_f = DataFactory()
        A_intensity, B_intensity = data_f.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)
        seq_data = A_intensity.iloc[:,2].to_frame()
        seq_data = seq_data.sort_values(by=seq_data.columns[0])
        v = (rankdata(seq_data, "average") / len(seq_data))
        v = pd.DataFrame(columns=["data"],data=v)

        dlist = {}
        for kurtosis_of_low_abundance in [0.1,0.6,1,2] :
            for saturation in [0.5,0.8,1]:
                res = data_f._clac_chance_to_be_zero(v, kurtosis_of_low_abundance=kurtosis_of_low_abundance, saturation=saturation)
                dlist[f"{kurtosis_of_low_abundance},{saturation}"] =  res["data"]


        # res = data_f._clac_chance_to_be_zero( v, kurtosis_of_low_abundance=0.99, saturation=0.7)

        final = pd.DataFrame(dlist)

        for col in final.columns :
            to_plot = final[col]
            plt.scatter(v.values,to_plot.values)
            plt.title(col)
            plt.show()
        self.fail()
