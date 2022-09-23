from unittest import TestCase
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
from sklearn import pipeline
from deconv_py.infras.data_factory import DataFactory
from deconv_py.infras.dashboards.deconvolution_results_plots import DeconvolutionResultsPlots as results_plots
import pandas as pd
import infras.cache_decorator as cache_decorator

class TestPpEntropyBased(TestCase):
    def test_transform(self):
        data_factory = DataFactory()
        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs("Intensity", index_func=lambda x: x.split(";")[0],
                                                          log2_transformation=True)

        _all_proteins_ind = pd.Index([])

        spec_cells, agg_spec_cells = PpKeepSpecificCells(), PpAggToSpecificCells()
        agg_iv, pp_irl_prot = AggregateIntraVariance(), PpCleanIrrelevantProteins()
        pp_chiv = PpCleanHighIntraVar()
        pp_entropy_only_largest, pp_entropy, pp_empty, pp_dep = PpEntropyBasedOnlyLargest(), PpEntropyBased(), PpEmpty(), PpDepDeBased()
        pp_svm_signature, pp_totel_sum = PpSvmSignature(), PpEntropyBasedTotelSum()
        pp_floor_quantile = PpFloorUnderQuantile()

        steps_few_cells = [("kepp_specific_cells", spec_cells), ("cleanHighIntraVariance", pp_chiv),
                           ("AggregateIntraVariance", agg_iv), ("cleen_irrelevant_proteins", pp_irl_prot),
                           ("PpEntropy", pp_entropy)]
        steps_all_cells = [("PpEmpty_cells_filt", pp_empty), ("cleanHighIntraVariance", pp_chiv),
                           ("AggregateIntraVariance", agg_iv), ("cleen_irrelevant_proteins", pp_irl_prot),
                           ("PpEntropy", pp_entropy)]

        _params = {"cleanHighIntraVariance__how": "std", "cleanHighIntraVariance__std_trh": 1,
                   "AggregateIntraVariance__how": "median", "PpEntropy__n_genes_per_cell": 10,
                   "PpEntropy__number_of_bins": 10, "PpEntropy__gene_entropy_trh": 1}

        # we dont want to remove all zeros genes in the mixtures - because thos is the problem we want to solve - so i replced B with A
        pip_all_cells = pipeline.Pipeline(steps=steps_all_cells)
        pip_all_cells.set_params(**_params)
        sig_all_cells = pip_all_cells.transform([A_all_vs, A_all_vs])
        _all_proteins_ind = _all_proteins_ind.union(sig_all_cells[0].index)

        # we dont want to remove all zeros genes in the mixtures - because thos is the problem we want to solve - so i replced B with A
        pip_few_cells = pipeline.Pipeline(steps=steps_few_cells)
        pip_few_cells.set_params(**_params)
        sig_few_cells = pip_few_cells.transform([A_all_vs, A_all_vs])
        all_proteins_ind = _all_proteins_ind.union(sig_few_cells[0].index)

        _all_proteins_ind = _all_proteins_ind.drop_duplicates()

