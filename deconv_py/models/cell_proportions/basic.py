from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
import pandas as pd
import scipy
from deconv_py.infras.cellMix.cellMix_coordinator import CellMixCoordinator
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator
import numpy as np
from models.cell_proportions.deconvolution_model import DeconvolutionModel

class BasicDeconv(DeconvolutionModel):
    def __init__(self, normalize=True, cellMix=False, em_optimisation=False, weight_sp=True,ensemble_learning = False):
        self.cellMix = cellMix
        self.cmc = CellMixCoordinator()

        DeconvolutionModel.__init__(self, normalize=normalize, em_optimisation=em_optimisation, weight_sp=weight_sp,ensemble_learning=ensemble_learning)

    def _deconvolution(self, mass_spec_mixture, cell_specific, weights=None):
        mass_spec_mixture, cell_specific = mass_spec_mixture.copy(deep=True), cell_specific.copy(deep=True)

        cell_abundance = scipy.optimize.nnls(cell_specific, mass_spec_mixture)[0]
        cell_abundance = pd.Series(data=cell_abundance, index=cell_specific.columns, name=mass_spec_mixture.name)

        return cell_abundance

        # region private
