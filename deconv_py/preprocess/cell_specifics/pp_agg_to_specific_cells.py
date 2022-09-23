from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from deconv_py.infras.global_utils import GlobalUtils
from infras.deconv_data_frame import DecovAccessor
import infras.cache_decorator as cache_decorator


class PpAggToSpecificCells(TransformerMixin,BaseEstimator):
    CYTOF_Mapping = _profilte_to_cytof_cell_map = {'NOT_BCellmemory': 'B cells ',
                                  'NOT_BCellnaive': 'B cells ',
                                  'NOT_BCellplasma': 'B cells ',
                                  'NOT_CD4TCellnaive': 'Na?ve CD4 Tcell',
                                  # 'NOT_CD4TCellTcm': 'CD4+central memory Tcells',
                                  'NOT_CD4TCellTem': 'CD4+ effector memory T cells',
                                  'NOT_CD4TCellTemra': 'CD4+ effector memory T cells',
                                  'NOT_CD8TCellnaive': 'Na?ve CD8 Tcell',
                                  'NOT_CD8TCellTem': 'CD8+ effector memory T cells',
                                  'NOT_CD8TCellTemra': 'CD8+ effector memory T cells',
                                  'NOT_DendriticCD1c': 'Plasmacytoid dendritic cells',
                                  'NOT_DendriticCD304': 'Plasmacytoid dendritic cells',
                                  'NOT_Monocytesclassical': 'Monocytes',
                                  'NOT_Monocytesintermediate': 'Monocytes',
                                  'NOT_Monocytesnonclassical': 'Monocytes',
                                  'NOT_NKCellsCD56bright': "NK"
                               }


    def __init__(self,cells_mapping = None):
        if cells_mapping is None :
            self.cells_mapping = self.CYTOF_Mapping
        else :
            self.cells_mapping = cells_mapping

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        A, B = data[0], data[1]
        A.deconv.set_agg_cells(True)

        relvent_cells = self.cells_mapping.keys()

        _A,_B = self._kepp_specific_cells_transformer([A,B],relvent_cells)

        mapping_low_level_cell = GlobalUtils.get_corospanding_cell_map_from_lists(_A.columns,relvent_cells)
        mapping_low_level_cell = {k:self.CYTOF_Mapping[v] for k,v in mapping_low_level_cell.items()}
        mapping_to_version = {cell:f"_0{cell.split('_0')[1]}" for cell in _A.columns}

        _A_res = _A.T.copy(deep = True)
        _A_res["low_level_cell"] = _A_res.index.map(lambda cell: mapping_low_level_cell[cell])
        _A_res["version"] = _A_res.index.map(lambda cell: mapping_to_version[cell])

        A_res = _A_res.groupby(by=["low_level_cell","version"]).sum()

        mapping_to_single_index = {ind:(ind[0]+ind[1]) for ind in A_res.index}
        A_res.index = A_res.index.map(lambda ind: mapping_to_single_index[ind])
        A_res = A_res.T

        try :
            A_res.deconv.transfer_all_relevant_properties(A)
            return [A_res,B]
        except Exception as  e :
            return [A_res,B]


    def _kepp_specific_cells_transformer(self,data,relvent_cells):
        A, B = data[0], data[1]

        mapping = GlobalUtils.get_corospanding_cell_map_from_lists(A.columns,relvent_cells)
        cols_to_keep = mapping.keys()
        A_res = A[cols_to_keep]

        return [A_res, B]

