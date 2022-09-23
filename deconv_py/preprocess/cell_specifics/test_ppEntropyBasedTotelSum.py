from unittest import TestCase
from deconv_py.infras.data_factory import DataFactory
from deconv_py.experiments.pipeline.pipeline_deconv import PipelineDeconv
from deconv_py.preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells
from deconv_py.preprocess.cell_specifics.pp_entropy_based_totel_sum import PpEntropyBasedTotelSum

class TestPpEntropyBasedTotelSum(TestCase):
    def test_transform(self):
        data_factory = DataFactory()
        agg_spec_cells = PpAggToSpecificCells()
        pp_ent_ts = PpEntropyBasedTotelSum()

        A_all_vs, B_all_vs = data_factory.load_IBD_all_vs(intensity_type="Intensity")

        _A, _B = agg_spec_cells.transform(data=[A_all_vs, B_all_vs ])
        res = pp_ent_ts.transform(data = [_A, _B])

