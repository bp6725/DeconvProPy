from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
import pandas as pd
import scipy
from models.cell_proportions.deconvolution_model import DeconvolutionModel
from sklearn.linear_model import ElasticNet
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator
import numpy as np


class RegressionDeconv(DeconvolutionModel):
    def __init__(self, normalize = True,regr_only_positive = True,regr_tol = 0.5,regr_alpha = 100,regr_normalize = True,l1_ratio = 0.5
                 ,em_optimisation = False,weight_sp = True,ensemble_learning=False):
        self.result = None

        self.regr_only_positive = regr_only_positive
        self.regr_tol = regr_tol
        self.regr_alpha = regr_alpha
        self.regr_normalize = regr_normalize
        self.l1_ratio = l1_ratio
        DeconvolutionModel.__init__(self, normalize=normalize, em_optimisation=em_optimisation, weight_sp=weight_sp,ensemble_learning=ensemble_learning)

    def _deconvolution(self,mass_spec_mixture, cell_specific, weights=None):
        mass_spec_mixture,cell_specific = mass_spec_mixture.copy(deep=True) ,cell_specific.copy(deep=True)

        regressor = ElasticNet(positive=self.regr_only_positive, tol=self.regr_tol,
                               alpha=self.regr_alpha, normalize=self.regr_normalize, l1_ratio=self.l1_ratio)
        regressor.fit(cell_specific, mass_spec_mixture)

        cell_abundance = pd.Series(data=regressor.coef_,index = cell_specific.columns,name = mass_spec_mixture.name)
        return cell_abundance

