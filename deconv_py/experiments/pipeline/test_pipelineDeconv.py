from unittest import TestCase
from experiments.pipeline.pipeline_deconv import PipelineDeconv
from preprocess.cell_specifics.pp_entropy_based_only_largest import PpEntropyBasedOnlyLargest
from preprocess.cell_specifics.pp_entropy_based import PpEntropyBased
from preprocess.cell_specifics.pp_empty import PpEmpty
from preprocess.cell_specifics.pp_clean_irrelevant_proteins import PpCleanIrrelevantProteins
from preprocess.cell_specifics.pp_clean_mostly_zeros_proteins import PpCleanMostlyZerosProteins
from preprocess.cell_specifics.pp_dep_de_based import PpDepDeBased
from preprocess.intra_variance.aggregate_intra_variance import AggregateIntraVariance
from measures.cell_proportions_measures.cell_proportions_measure import CellProportionsMeasure
from preprocess.intra_variance.pp_clean_high_intra_var import PpCleanHighIntraVar
from preprocess.cell_specifics.pp_keep_specific_cells import PpKeepSpecificCells
from preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells
from preprocess.cell_specifics.pp_svm_signature import PpSvmSignature
from preprocess.cell_specifics.pp_entropy_based_totel_sum import PpEntropyBasedTotelSum
from preprocess.cell_specifics.pp_floor_under_quantile import PpFloorUnderQuantile
from preprocess.normalization.signature_normalization import SignatureNormalization

from models.cell_proportions.basic import BasicDeconv
from models.cell_proportions.regression import RegressionDeconv
from models.cell_proportions.generalized_estimating_equations import GeneralizedEstimatingEquations
from models.cell_proportions.robust_linear_model import RobustLinearModel
from models.cell_proportions.ransac_model import RansacModel
from models.cell_proportions.test import Test

from infras.data_factory import DataFactory
from infras.dashboards.deconvolution_results_plots import DeconvolutionResultsPlots as results_plots
from measures.hyperparameter_measure import HyperParameterMeasures

import pickle as pkl
import numpy as np
import pandas as pd
from deconv_py.infras.global_utils import GlobalUtils
import matplotlib.pyplot as plt

class TestPipelineDeconv(TestCase):
    def test_run_cytof_pipeline(self):
        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot,pp_clean_mz_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins(),PpCleanMostlyZerosProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd = BasicDeconv()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "floor_quantile", "function": pp_floor_quantile,
                  "params": {'quantile' : [0.3,0.5,0.7]}},
                 # {"function_name": "PpEmpty_floor", "function": pp_empty,
                 #  "params": {}}
             ]},
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                  "params": {}},
                 # {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                 #  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                  # "params": {"how": ["std"], "std_trh": [1, 2, 4]}},
                  "params": {"how": ["std"], "std_trh": [1]}},
                 # {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                 #  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  # "params": {"how": ["mean", "median", "max"]}}]},
                                            "params": {"how": ["median"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "PpCleanMostlyZerosProteins", "function": pp_clean_mz_prot,
                  "params": {}},
                 # {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                 #  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 # {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                 #  "params": {"totel_sum_percentage": [0.001, 0.01], "with_norm": [False],"number_of_bins" :[0,10,20] ,
                 #             "only_largest": [True, False]}},
                 # # {"function_name": "PpEntropyBased", "function": pp_entropy,
                 # #  "params": {"n_genes_per_cell": [20, 100], "gene_entropy_trh": [1, 2, 3],"number_of_bins" :[0,10,20] ,
                 # #             "with_norm": [False]}},
                 # {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,"number_of_bins" :[0,10,20] ,
                 #  # "params": {"n_genes_per_cell": [10, 80], "with_norm": [False]}},
                 #  "params": {"n_genes_per_cell": [10], "with_norm": [False]}},
                 {"function_name": "PpDepDeBased", "function": pp_dep,
                  # "params": {"n_of_genes": [10, 80], "is_agg_cells": [True, False]}},
                  "params": {"n_of_genes": [10], "is_agg_cells": [True, False]}},
                 # {"function_name": "PpSvm", "function": pp_svm_signature,
                 #  "params": {"n_features": [10, 80], "with_norm": [False]}},
                 # {"function_name": "PpEmpty_prepro", "function": pp_empty,
                 #  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                 {"function_name": "BasicDeconv", "function": bd,
                  "params": {"normalize": [True,False], "cellMix": [False]}}]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [True], "with_iso_test": [True]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("iBAQ", index_func=lambda x: x, log2_transformation=True)

        results = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs, per_cell_analysis=False,with_cache=False)

        results_plots.describe_results(results.iloc[1]["uuid"], results)

        print(results)
        print("finish")

    def test_run_pipeline(self):
        data_factory = DataFactory()
        A_all_vs, _ = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                   log2_transformation=True)
        _, B_am, X = data_factory.load_simple_artificial_profile("Intensity", index_func=lambda x: x.split(";")[0],
                                                                 log2_transformation=True)

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd = BasicDeconv()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [{"step_name": "floor",
                                "steps": [
                                    {"function_name": "floor_quantile", "function": pp_floor_quantile,
                                     "params": {}},
                                    {"function_name": "PpEmpty_floor", "function": pp_empty,
                                     "params": {}}
                                ]},
                               #                        -----------------------------------
                               {"step_name": "per_cells_filter",
                                "steps": [
                                    {"function_name": "kepp_specific_cells", "function": spec_cells,
                                     "params": {"cells_list":[['Intensity NOT_CD4TCellTcm','Intensity NOT_BCellmemory','Intensity NOT_Monocytesnonclassical']]}}
                                ]},
                               # -------------------------------
                               {"step_name": "cleanHighIntraVariance",
                                "steps": [
                                    {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                                     "params": {"how": ["std"], "std_trh": [1, 2]}},
                                    {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                                     "params": {}}]},
                               # -------------------------------
                               {"step_name": "AggregateIntraVariance",
                                "steps": [
                                    {"function_name": "AggregateIntraVariance", "function": agg_iv,
                                     "params": {"how": ["mean", "median", "max"]}}]},
                               #                                 "params": {"how": ["mean"]}}]},
                               # --------------------------------
                               {"step_name": "cleen_irrelevant_proteins",
                                "steps": [
                                    {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                                     "params": {}}]},
                               # --------------------------------
                               {"step_name": "Cytof_X_Building",
                                "steps": [
                                    {"function_name": "Cytof_X_Building", "function": pp_empty,
                                     "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
                               # --------------------------------
                               {"step_name": "preprocess",
                                "steps": [
                                    {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                                     "params": {"totel_sum_percentage": [0.001, 0.0001], "with_norm": [True, False],
                                                "number_of_bins": [0, 10, 20],
                                                #                             "params": {"totel_sum_percentage": [0.001, 0.0001],"with_norm": [False],"number_of_bins" :[0,20] ,
                                                "only_largest": [True, False]}},
                                    {"function_name": "PpEntropyBased", "function": pp_entropy,
                                     "params": {"n_genes_per_cell": [20, 100], "gene_entropy_trh": [1, 3],
                                                "number_of_bins": [0, 10, 20],
                                                "with_norm": [True, False]}},
                                    {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,
                                     "params": {"n_genes_per_cell": [20, 80], "number_of_bins": [0, 10, 20],
                                                "with_norm": [True, False]}},
                                    {"function_name": "PpDepDeBased", "function": pp_dep,
                                     "params": {"n_of_genes": [20, 80], "is_agg_cells": [True, False]}},
                                    {"function_name": "PpSvm", "function": pp_svm_signature,
                                     "params": {"n_features": [20, 80], "with_norm": [False]}},
                                    {"function_name": "PpEmpty_prepro", "function": pp_empty,
                                     "params": {}}
                                ]},
                               # --------------------------------
                               {"step_name": "deconv",
                                "steps": [
                                    {"function_name": "BasicDeconv", "function": bd,
                                     "params": {"normalize": [True], "cellMix": [False]}}]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [True], "with_iso_test": [True]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results = _pipe.run_pipeline(A_all_vs, B_am, X)
        print(meta_results )

    def test_full_cytof_pipeline(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)
        # A_all_vs_not_impu, B_all_vs_not_impu = data_factory.load_no_imputation_IBD_all_vs("Intensity",
        #                                                                                   index_func=lambda x:
        #                                                                                   x.split(";")[0],
        #                                                                                   log2_transformation=False)
        # B_all_vs_not_impu = B_all_vs_not_impu.replace('Filtered', 0)
        # B_all_vs_not_impu = B_all_vs_not_impu.astype(float)
        # B_all_vs_not_impu = B_all_vs_not_impu[B_all_vs.columns.intersection(B_all_vs_not_impu.columns)]

        # pick_set = PickDataSet()
        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd,rd = BasicDeconv(),RegressionDeconv()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "floor_quantile", "function": pp_floor_quantile,
                  "params": {}},
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                  {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                      "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                  #                              "params": {"how": ["std"],"std_trh":[1,2]}},
                  "params": {"how": ["std"], "std_trh": [1,4]}},
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": ["mean", "median"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                  #                     "params": {"totel_sum_percentage": [0.001, 0.0001],"with_norm": [True,False],"number_of_bins" :[0,10,20] ,
                  "params": {"totel_sum_percentage": [0.001, 0.0001], "with_norm": [False,True], "number_of_bins": [0, 20],
                             "only_largest": [True, False]}},
                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [20, 100,250], "gene_entropy_trh": [1, 3], "number_of_bins": [0, 20],
                             "with_norm": [True, False]}},
                 {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,
                  #                              "params": {"n_genes_per_cell": [20,80],"number_of_bins" :[0,10,20] ,"with_norm": [True, False]}},
                  "params": {"n_genes_per_cell": [20, 80,250], "number_of_bins": [0, 20], "with_norm": [True, False]}},
                 # {"function_name": "PpDepDeBased", "function": pp_dep,
                 #  "params": {"n_of_genes": [20, 80], "is_agg_cells": [True, False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [20, 80,250], "with_norm": [False,True]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                 {"function_name": "RegressionDeconv", "function": rd,
                  "params": {"normalize": [True]}},
                 {"function_name": "BasicDeconv", "function": bd,
                  "params": {"normalize": [True], "cellMix": [False]}}
             ]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "MI", "entropy"], "with_pvalue": [False],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs, per_cell_analysis=False)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data,measure_trh=0.25,
                                                        feature_columns=meta_results_original_data.columns.difference(
                                                            pd.Index(["corrMean", "MIMean", "entropy","uuid"])).to_list())

        # plt.show()
        # meta_results_not_imputed = _pipe.run_cytof_pipeline(A_all_vs_not_impu, B_all_vs_not_impu,
        #                                                     per_cell_analysis=False,with_cache=True,cache_specific_signature = "not_imputed " )
        # HyperParameterMeasures.plot_hyperparameter_tree(meta_results_not_imputed,measure_trh=0.25,
        #                                                 feature_columns=meta_results_not_imputed.columns.difference(
        #                                                     pd.Index(["corrMean", "MIMean", "entropy","uuid"])).to_list())

        print("finish")

    def test_full_cytof_pipline_on_simulation(self):
        data_factory = DataFactory()
        A_all_vs, _ = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)
        tmp,X,B_all_vs = data_factory.build_simulated_data(percantage_to_zero=0.01, kurtosis_of_low_abundance=0.8, saturation=0.95,
                                          unquantified_cell_percentage=5)

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        sig_norm = SignatureNormalization()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd = BasicDeconv()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "floor_quantile", "function": pp_floor_quantile,
                  "params": {}},
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                  "params": {}},
                 #                          {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                 #                              "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                  #                              "params": {"how": ["std"],"std_trh":[1,2]}},
                  "params": {"how": ["std"], "std_trh": [1]}},
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": ["mean", "median"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                  #                     "params": {"totel_sum_percentage": [0.001, 0.0001],"with_norm": [True,False],"number_of_bins" :[0,10,20] ,
                  "params": {"totel_sum_percentage": [0.001, 0.0001], "with_norm": [False], "number_of_bins": [0, 20],
                             "only_largest": [True, False]}},
                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [20, 100], "gene_entropy_trh": [1, 3], "number_of_bins": [0, 20],
                             "with_norm": [True, False]}},
                 {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,
                  #                              "params": {"n_genes_per_cell": [20,80],"number_of_bins" :[0,10,20] ,"with_norm": [True, False]}},
                  "params": {"n_genes_per_cell": [20, 80], "number_of_bins": [0, 20], "with_norm": [True, False]}},
                 # {"function_name": "PpDepDeBased", "function": pp_dep,
                 #  "params": {"n_of_genes": [20, 80], "is_agg_cells": [True, False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [20, 80], "with_norm": [False]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                 {"function_name": "BasicDeconv", "function": bd,
                  "params": {"normalize": [True], "cellMix": [False]}}]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation"], "with_pvalue": [False],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_pipeline(A_all_vs, B_all_vs,X)
        # meta_results_not_imputed = _pipe.run_cytof_pipeline(A_all_vs_not_impu, B_all_vs_not_impu,
        #                                                     per_cell_analysis=False)

        print("finish")

    def test_full_cytof_pipline_on_naive_simulation(self):
        '''
        for naive simulation :
        - we take original A
        - then we agg so we have only one v per cell
        - then we generete X from simulation class
        - then we build B = A*X
        - then we add small error to create new v's
        :return:
        '''
        data_factory = DataFactory()
        A_org, _ = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)
        _, X, _ = data_factory.build_simulated_data(percantage_to_zero=0.01, kurtosis_of_low_abundance=0.8,
                                                             saturation=0.95,
                                                             unquantified_cell_percentage=5)

        agg_iv= AggregateIntraVariance()
        A,_ = agg_iv.transform([A_org,None])

        A_vs_list = []
        for i in range(1,4) :
            error_mtx = np.ones(A.shape) + np.random.random(A.shape)/10
            Avs = A.copy(deep=True) * error_mtx
            Avs = Avs.rename(columns = {col:f"{col}_0{i}" for col in  Avs.columns})
            A_vs_list.append(Avs.copy(deep=True))

        A_all_vs= pd.concat(A_vs_list, axis=1)
        A_all_vs.columns.name = None

        X = X.rename(index={i: f"Intensity NOT_{i}" for i in X.index})
        mutual_cells = A.columns.intersection(X.index)

        B_all_vs = A[mutual_cells].dot(X.loc[mutual_cells])


        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd = BasicDeconv()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "floor_quantile", "function": pp_floor_quantile,
                  "params": {}},
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                 {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 # {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                 #  #                              "params": {"how": ["std"],"std_trh":[1,2]}},
                 #  "params": {"how": ["std"], "std_trh": [1]}},
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": [ "median"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                  #                     "params": {"totel_sum_percentage": [0.001, 0.0001],"with_norm": [True,False],"number_of_bins" :[0,10,20] ,
                  "params": {"totel_sum_percentage": [0.001], "with_norm": [False,True], "number_of_bins": [0],
                             "only_largest": [True, False]}},
                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [20], "gene_entropy_trh": [1], "number_of_bins": [ 20],
                             "with_norm": [True, False]}},
                 {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,
                  #                              "params": {"n_genes_per_cell": [20,80],"number_of_bins" :[0,10,20] ,"with_norm": [True, False]}},
                  "params": {"n_genes_per_cell": [20], "number_of_bins": [10,20], "with_norm": [True, False]}},
                 # {"function_name": "PpDepDeBased", "function": pp_dep,
                 #  "params": {"n_of_genes": [20, 80], "is_agg_cells": [True, False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [20], "with_norm": [False]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                 {"function_name": "BasicDeconv", "function": bd,
                  "params": {"normalize": [True,False], "cellMix": [False]}}]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation"], "with_pvalue": [False],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_pipeline(A_all_vs, B_all_vs,X)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data)
        # meta_results_not_imputed = _pipe.run_cytof_pipeline(A_all_vs_not_impu, B_all_vs_not_impu,
        #                                                     per_cell_analysis=False)

        print("finish")

    def test_full_cytof_pipeline_on_regr_opti(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)

        # pick_set = PickDataSet()
        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        bd, rd = BasicDeconv(), RegressionDeconv()
        gee,rlm,rsm = GeneralizedEstimatingEquations(),RobustLinearModel(),RansacModel()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "floor_quantile", "function": pp_floor_quantile,
                  "params": {}},
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                 {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                  #                              "params": {"how": ["std"],"std_trh":[1,2]}},
                  "params": {"how": ["std"], "std_trh": [1,2]}},
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": [ "median"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1], "number_of_bins": [0],
                             "with_norm": [False]}},
                 {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,
                  #                              "params": {"n_genes_per_cell": [20,80],"number_of_bins" :[0,10,20] ,"with_norm": [True, False]}},
                  "params": {"n_genes_per_cell": [20,100], "number_of_bins": [0], "with_norm": [False]}},
                 # {"function_name": "PpDepDeBased", "function": pp_dep,
                 #  "params": {"n_of_genes": [20, 80], "is_agg_cells": [True, False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [20,100], "with_norm": [False]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                 # {"function_name": "GEEDeconv", "function": gee,
                 #  "params": {"normalize": [True], "cellMix": [False]}},
                 {"function_name": "Ransac", "function": rsm,
                  "params": {"normalize": [True], "cellMix": [False]}},
                 {"function_name": "RLMDeconv", "function": rlm,
                  "params": {"normalize": [True], "cellMix": [False]}}
             ]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [False],"correlation_method" : ["pearson","spearman"],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs,cache_specific_signature="few_regr")
        meta_results_original_data["corrMean"] = meta_results_original_data["corrMean"].fillna(-1)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data, measure_trh=0.4,
                                                        feature_columns=meta_results_original_data.columns.difference(
                                                            pd.Index(
                                                                ["corrMean", "MIMean", "entropy", "uuid"])).to_list())

        print("finish")

    def test_full_cytof_pipeline_on_EM_opti(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()

        bd,rd,rlm,gee = BasicDeconv(),RegressionDeconv(),RobustLinearModel(),GeneralizedEstimatingEquations()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                 {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": ["median","max"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 # {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                 #  "params": {"totel_sum_percentage": [0.01, 0.001], "with_norm": [False, True],
                 #             "number_of_bins": [0, 20],
                 #             "only_largest": [True, False]}},
                 {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                  "params": {"totel_sum_percentage": [0.01], "with_norm": [False, True],
                             "number_of_bins": [20],
                             "only_largest": [True]}},

                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [100], "gene_entropy_trh": [1], "number_of_bins": [20],
                             "with_norm": [False,True]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  # "params": {"n_features": [40,100], "with_norm": [False,True]}},
                 "params": {"n_features": [100], "with_norm": [False, True]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                {"function_name": "BasicDeconv", "function": bd,
                "params": {'em_optimisation':[True,False],"weight_sp":[True,False]}},
                {"function_name": "RegressionDeconv", "function": rd,
                 "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
                {"function_name": "RobustLinearDeconv", "function": rlm,
                 "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
                 # {"function_name": "GeneralizedEstimatingDeconv", "function": gee,
                 #  "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
             ]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [False],
                             "correlation_method": ["pearson"],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs,per_cell_analysis=False,with_cache=True,cache_specific_signature="with_em_with_A_imputation_corrected_test")
        meta_results_original_data["corrMean"] = meta_results_original_data["corrMean"].fillna(-1)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data, measure_trh=0.4,
                                                        feature_columns=meta_results_original_data.columns.difference(
                                                            pd.Index(
                                                                ["corrMean", "MIMean", "entropy", "uuid"])).to_list())

        # plt.show()
        # meta_results_not_imputed = _pipe.run_cytof_pipeline(A_all_vs_not_impu, B_all_vs_not_impu,
        #                                                     per_cell_analysis=False,with_cache=True,cache_specific_signature = "not_imputed " )
        # HyperParameterMeasures.plot_hyperparameter_tree(meta_results_not_imputed,measure_trh=0.25,
        #                                                 feature_columns=meta_results_not_imputed.columns.difference(
        #                                                     pd.Index(["corrMean", "MIMean", "entropy","uuid"])).to_list())

        print("finish")

    def test_full_cytof_pipeline_on_weak_learners(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        pp_chiv = PpCleanHighIntraVar()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        sig_norm = SignatureNormalization()

        bd,rd,rlm,gee = BasicDeconv(),RegressionDeconv(),RobustLinearModel(),GeneralizedEstimatingEquations()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {"cells_list":[['NOT_BCellmemory','NOT_BCellnaive','NOT_CD4TCellTcm','NOT_CD4TCellnTregs',
                  'NOT_CD4TCellnaive','NOT_CD8TCellnaive','NOT_Monocytesclassical','NOT_Monocytesintermediate',
                  'NOT_Monocytesnonclassical','NOT_NKCellsCD56bright']]}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                 {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 # {"function_name": "PpCleanHighIntraVar", "function": pp_chiv,
                 #  "params": {"how": ["std"], "std_trh": [1]}},
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": ["mean","median","max"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  "params": {"n_genes_per_cell": [200], "gene_entropy_trh": [1], "number_of_bins": [20],
                             "with_norm": [False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [200], "with_norm": [False]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "signature_normalization",
             "steps": [
                 {"function_name": "sig_norm", "function": sig_norm,
                  "params": {"normalization_strategy":["mean","max"]}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                {"function_name": "BasicDeconv", "function": bd,
                "params": {'em_optimisation':[False],"weight_sp":[False],"ensemble_learning":[True]}},
                {"function_name": "RegressionDeconv", "function": rd,
                 "params": {'em_optimisation': [False], "weight_sp": [False],"ensemble_learning":[True]}},
                {"function_name": "RobustLinearDeconv", "function": rlm,
                 "params": {'em_optimisation': [False], "weight_sp": [False],"ensemble_learning":[True]}},
             ]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [False],
                             "correlation_method": ["pearson"],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs,per_cell_analysis=True,with_cache=True,
                                                              with_tree_cache = True,cache_specific_signature="weak_learnears")
        meta_results_original_data["corrMean"] = meta_results_original_data["corrMean"].fillna(-1)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data, measure_trh=0.5,
                                                        feature_columns=meta_results_original_data.columns.difference(
                                                            pd.Index(
                                                                ["corrMean", "MIMean", "entropy", "uuid"])).to_list())


        print("finish")

    def test_full_new_cytof(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()

        bd,rd,rlm,gee = BasicDeconv(),RegressionDeconv(),RobustLinearModel(),GeneralizedEstimatingEquations()
        cpm = CellProportionsMeasure(how="groups")

        hyper_configuration = [
            {"step_name": "floor",
             "steps": [
                 {"function_name": "PpEmpty_floor", "function": pp_empty,
                  "params": {}}
             ]},
            #                        -----------------------------------
            {"step_name": "per_cells_filter",
             "steps": [
                 {"function_name": "kepp_specific_cells", "function": spec_cells,
                  "params": {}},
                 # {"function_name": "agg_to_specific_cells", "function": agg_spec_cells,
                 #  "params": {}},
                 {"function_name": "PpEmpty_cells_filt", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "cleanHighIntraVariance",
             "steps": [
                 {"function_name": "PpEmpty_clean_iv", "function": pp_empty,
                  "params": {}}
             ]},
            # -------------------------------
            {"step_name": "AggregateIntraVariance",
             "steps": [
                 {"function_name": "AggregateIntraVariance", "function": agg_iv,
                  #                              "params": {"how": ["mean", "median","max"]}}]},
                  "params": {"how": ["median","max"]}}]},
            # --------------------------------
            {"step_name": "cleen_irrelevant_proteins",
             "steps": [
                 {"function_name": "CleanIrrelevantProteins", "function": pp_irl_prot,
                  "params": {}}]},
            # --------------------------------
            {"step_name": "Cytof_X_Building",
             "steps": [
                 {"function_name": "Cytof_X_Building", "function": pp_empty,
                  "params": {"keep_labels": [True], "with_label_prop": [False]}}]},
            # --------------------------------
            {"step_name": "preprocess",
             "steps": [
                 # {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                 #  "params": {"totel_sum_percentage": [0.01, 0.001], "with_norm": [False, True],
                 #             "number_of_bins": [0, 20],
                 #             "only_largest": [True, False]}},
                 {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                  "params": {"totel_sum_percentage": [0.01], "with_norm": [False, True],
                             "number_of_bins": [20],
                             "only_largest": [True]}},

                 {"function_name": "PpEntropyBased", "function": pp_entropy,
                  #                              "params": {"n_genes_per_cell": [20,100], "gene_entropy_trh": [1,3],"number_of_bins" :[0,10,20] ,
                  "params": {"n_genes_per_cell": [100], "gene_entropy_trh": [1], "number_of_bins": [20],
                             "with_norm": [False,True]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  # "params": {"n_features": [40,100], "with_norm": [False,True]}},
                 "params": {"n_features": [100], "with_norm": [False, True]}},
                 {"function_name": "PpEmpty_prepro", "function": pp_empty,
                  "params": {}}
             ]},
            # --------------------------------
            {"step_name": "deconv",
             "steps": [
                {"function_name": "BasicDeconv", "function": bd,
                "params": {'em_optimisation':[True,False],"weight_sp":[True,False]}},
                {"function_name": "RegressionDeconv", "function": rd,
                 "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
                {"function_name": "RobustLinearDeconv", "function": rlm,
                 "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
                 # {"function_name": "GeneralizedEstimatingDeconv", "function": gee,
                 #  "params": {'em_optimisation': [True,False], "weight_sp": [True,False]}},
             ]}]

        hyper_measure_configuration = [
            {"step_name": "measure",
             "steps": [
                 {"function_name": "CellProportionsMeasure", "function": cpm,
                  #           "params": {"how": ["correlation","RMSE","MI"],"with_pvalue":[True],"with_iso_test":[False]}}]}]
                  "params": {"how": ["correlation", "entropy"], "with_pvalue": [False],
                             "correlation_method": ["pearson"],
                             "with_iso_test": [False]}}]}]

        _pipe = PipelineDeconv(hyper_configuration=hyper_configuration,
                               hyper_measure_configuration=hyper_measure_configuration)

        meta_results_original_data = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs,per_cell_analysis=False,
                                                              with_cache=True,
                                                              cache_specific_signature="new_cytof_test1")
        meta_results_original_data["corrMean"] = meta_results_original_data["corrMean"].fillna(-1)
        HyperParameterMeasures.plot_hyperparameter_tree(meta_results_original_data, measure_trh=0.4,
                                                        feature_columns=meta_results_original_data.columns.difference(
                                                            pd.Index(
                                                                ["corrMean", "MIMean", "entropy", "uuid"])).to_list())

        # plt.show()
        # meta_results_not_imputed = _pipe.run_cytof_pipeline(A_all_vs_not_impu, B_all_vs_not_impu,
        #                                                     per_cell_analysis=False,with_cache=True,cache_specific_signature = "not_imputed " )
        # HyperParameterMeasures.plot_hyperparameter_tree(meta_results_not_imputed,measure_trh=0.25,
        #                                                 feature_columns=meta_results_not_imputed.columns.difference(
        #                                                     pd.Index(["corrMean", "MIMean", "entropy","uuid"])).to_list())

        print("finish")
