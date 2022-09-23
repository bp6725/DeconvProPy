from unittest import TestCase
from deconv_py.experiments.pipeline.pipeline_deconv import PipelineDeconv
from deconv_py.preprocess.cell_specifics.pp_entropy_based_only_largest import PpEntropyBasedOnlyLargest
from deconv_py.preprocess.cell_specifics.pp_entropy_based import PpEntropyBased
from deconv_py.preprocess.cell_specifics.pp_empty import PpEmpty
from deconv_py.preprocess.cell_specifics.pp_clean_irrelevant_proteins import PpCleanIrrelevantProteins
from deconv_py.preprocess.cell_specifics.pp_dep_de_based import PpDepDeBased
from deconv_py.models.cell_proportions.basic import BasicDeconv
from deconv_py.preprocess.intra_variance.aggregate_intra_variance import AggregateIntraVariance
from deconv_py.measures.cell_proportions_measures.cell_proportions_measure import CellProportionsMeasure
from deconv_py.preprocess.intra_variance.pp_clean_high_intra_var import PpCleanHighIntraVar
from deconv_py.preprocess.cell_specifics.pp_keep_specific_cells import PpKeepSpecificCells
from deconv_py.preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells
from deconv_py.preprocess.cell_specifics.pp_svm_signature import PpSvmSignature
from deconv_py.preprocess.cell_specifics.pp_entropy_based_totel_sum import PpEntropyBasedTotelSum
from deconv_py.preprocess.cell_specifics.pp_floor_under_quantile import PpFloorUnderQuantile
from deconv_py.infras.data_factory import DataFactory
from deconv_py.infras.dashboards.deconvolution_results_plots import DeconvolutionResultsPlots as results_plots
import pickle as pkl


class TestPipelineDeconv(TestCase):
    def test_run_cytof_pipeline(self):
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
            {"step_name": "per_cells_filter",
             "steps": [
                 # {"function_name": "kepp_specific_cells", "function": spec_cells,
                 #  "params": {}},
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
                 # {"function_name": "pp_totel_sum", "function": pp_totel_sum,
                 #  "params": {"totel_sum_percentage": [0.001, 0.01], "with_norm": [False],"number_of_bins" =[0,10,20] ,
                 #             "only_largest": [True, False]}},
                 # {"function_name": "PpEntropyBased", "function": pp_entropy,
                 #  "params": {"n_genes_per_cell": [20, 100], "gene_entropy_trh": [1, 2, 3],"number_of_bins" =[0,10,20] ,
                 #             "with_norm": [False]}},
                 # {"function_name": "PpEntropyBasedOnlyLargest", "function": pp_entropy_only_largest,"number_of_bins" =[0,10,20] ,
                 #  "params": {"n_genes_per_cell": [10, 80], "with_norm": [False]}},
                 # {"function_name": "PpDepDeBased", "function": pp_dep,
                 #  "params": {"n_of_genes": [10, 80], "is_agg_cells": [True, False]}},
                 {"function_name": "PpSvm", "function": pp_svm_signature,
                  "params": {"n_features": [10, 80], "with_norm": [False]}}
                 #                             {"function_name": "PpEmpty_prepro", "function": pp_empty,
                 #                              "params": {}}
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

        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("iBAQ", index_func=lambda x: x, log2_transformation=True)

        results = _pipe.run_cytof_pipeline(A_all_vs, B_all_vs, per_cell_analysis=False)

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
