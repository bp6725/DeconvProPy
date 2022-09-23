from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from deconv_py.infras.global_utils import GlobalUtils

class PpKeepSpecificCells(TransformerMixin,BaseEstimator):
    CYTOF_CELLS = _profilte_to_cytof_cell_map = ['NOT_BCellmemory','NOT_BCellnaive','NOT_BCellplasma','NOT_CD4TCellnaive',
                                  'NOT_CD4TCellnTregs','NOT_CD4TCellTcm','NOT_CD4TCellTem','NOT_CD4TCellTemra','NOT_CD8TCellnaive',
                                  'NOT_CD8TCellTem','NOT_CD8TCellTemra','NOT_DendriticCD1c','NOT_DendriticCD304','NOT_Monocytesclassical',
                                                 'NOT_Monocytesintermediate','NOT_Monocytesnonclassical']


    def __init__(self,cells_list = None):
        if cells_list is None :
            self.cells_list = self.CYTOF_CELLS
        else :
            self.cells_list = cells_list


    def transform(self, data, *_):
        A, B = data[0], data[1]

        mapping = GlobalUtils.get_corospanding_cell_map_from_lists(A.columns,self.cells_list)
        cols_to_keep = mapping.keys()

        A_res = A[cols_to_keep]


        try :
            A_res.deconv.transfer_all_relevant_properties(A)
            return [A_res,B]
        except :
            return [A_res,B]
