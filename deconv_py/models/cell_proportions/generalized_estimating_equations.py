from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
import pandas as pd
import scipy
from deconv_py.infras.cellMix.cellMix_coordinator import CellMixCoordinator
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import os
import pickle as pkl
from models.cell_proportions.deconvolution_model import DeconvolutionModel

class GeneralizedEstimatingEquations(DeconvolutionModel):
    def __init__(self, normalize = True,cellMix = False,em_optimisation = False,weight_sp = True,ensemble_learning = False):
        DeconvolutionModel.__init__(self, normalize=normalize, em_optimisation=em_optimisation, weight_sp=weight_sp,ensemble_learning=ensemble_learning)

    def _read_original_labels(self, full_mass_spec_data, corr_from_cache = False):
        if corr_from_cache :
            path = r"C:\Repos\deconv_py\deconv_py\cache\group_label_cache.pkl"
            if os.path.exists(path) :
                idx_to_group = pd.read_pickle(path)
                return idx_to_group

        genes_corr_df = full_mass_spec_data.T.corr().copy(deep=True).fillna(0)
        dissimilarity = 1 - (genes_corr_df)
        np.fill_diagonal(dissimilarity.values, 0)
        dissimilarity = scipy.clip(dissimilarity, 0, 2)
        hierarchy = linkage(squareform(dissimilarity), method='average')
        labels = fcluster(hierarchy, 0.5, criterion='distance')
        idx_to_group = {str(idx): lab for lab, idx in zip(labels, full_mass_spec_data.index)}

        if corr_from_cache :
            pd.to_pickle(idx_to_group,path)

        return idx_to_group

    def _set_genes_to_groups(self,original_mass_spec_data,mass_spec:pd.Series ):
        idx_to_group =  self._read_original_labels(original_mass_spec_data,True)

        return [idx_to_group[str(g)] for g in mass_spec.index]

    def _deconvolution(self,mass_spec_mixture, cell_specific_, weights=None):
        mass_spec_mixture,cell_specific = mass_spec_mixture.copy(deep=True) ,cell_specific_.copy(deep=True)
        groups_list = self._set_genes_to_groups(cell_specific_.deconv.original_data[0],mass_spec_mixture)

        model = sm.GEE(mass_spec_mixture, cell_specific, groups=groups_list)
        result = model.fit(maxiter=600)

        cell_abundance = result.params
        cell_abundance[cell_abundance<0] = 0
        cell_abundance = cell_abundance._set_name(mass_spec_mixture.name)
        return cell_abundance

