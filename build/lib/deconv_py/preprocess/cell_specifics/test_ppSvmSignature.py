from unittest import TestCase
from deconv_py.infras.data_factory import DataFactory
from deconv_py.experiments.pipeline.pipeline_deconv import PipelineDeconv
from deconv_py.preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells

class TestPpSvmSignature(TestCase):
    def test_transform(self):
        _pipe = PipelineDeconv({},{})
        data_factory = DataFactory()
        agg_spec_cells = PpAggToSpecificCells()

        best_results_and_known = _pipe.load_results_from_archive("3113972946")
        _, simx, _ = data_factory.build_simulated_data()
        simx.index = simx.index.map(lambda x: f"NOT_{x}_01")

        _simx, _ = agg_spec_cells.transform(data = [simx.T, simx.T])
