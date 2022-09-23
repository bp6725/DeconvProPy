from unittest import TestCase
from deconv_py.infras.dashboards.deconvolution_results_plots import DeconvolutionResultsPlots
from deconv_py.preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells

class TestDeconvolutionResultsPlots(TestCase):
    def test_describe_results(self):
        DeconvolutionResultsPlots.describe_results("1568926666")
