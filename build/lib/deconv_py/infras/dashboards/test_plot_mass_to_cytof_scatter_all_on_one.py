from unittest import TestCase
import deconv_py.infras.dashboards.exploration_cytof_plots as plots
from deconv_py.experiments.pipeline.pipeline_deconv import PipelineDeconv
from infras.global_utils import GlobalUtils

class TestPlot_mass_to_cytof_scatter_all_on_one(TestCase):
    def test_plot_mass_to_cytof_scatter_all_on_one(self):
        _pipe = PipelineDeconv(hyper_configuration=[],
                               hyper_measure_configuration=[])
        best_results_and_known = _pipe.load_results_from_archive("2814562996")

        best_results = best_results_and_known["result"]
        best_known = best_results_and_known["known"]
        best_known = best_known.rename(columns=GlobalUtils.get_corospanding_mixtures_map(best_known, best_results))

        best_known = best_known.drop(columns=['46_v2', '33_v2', '40_v2', '39_v3'])

        plots.plot_mass_to_cytof_scatter_all_on_one(best_results, best_known, best_results)

