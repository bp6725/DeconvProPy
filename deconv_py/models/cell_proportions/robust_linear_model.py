from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import pandas as pd
import scipy
from deconv_py.infras.cellMix.cellMix_coordinator import CellMixCoordinator
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
from models.cell_proportions.deconvolution_model import DeconvolutionModel


class RobustLinearModel(DeconvolutionModel):
    def __init__(self, normalize=True, cellMix=False, em_optimisation=False, weight_sp=True,ensemble_learning = False):
        DeconvolutionModel.__init__(self, normalize=normalize, em_optimisation=em_optimisation, weight_sp=weight_sp,ensemble_learning=ensemble_learning)

    def _deconvolution(self, mass_spec_mixture, cell_specific, weights=None):
        mass_spec_mixture, cell_specific = mass_spec_mixture.copy(deep=True), cell_specific.copy(deep=True)

        rlm_model = sm.RLM(mass_spec_mixture, cell_specific, M=sm.robust.norms.HuberT())

        result = rlm_model.fit()
        cell_abundance = result.params
        cell_abundance[cell_abundance < 0] = 0
        cell_abundance = cell_abundance._set_name(mass_spec_mixture.name)

        return cell_abundance
