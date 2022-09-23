from models.cell_proportions.deconvolution_model import DeconvolutionModel
import numpy as np
from sklearn import linear_model
import pandas as pd


class RansacModel(DeconvolutionModel):
    def __init__(self, normalize=True, cellMix=False, em_optimisation=False, weight_sp=True,ensemble_learning = False):
        DeconvolutionModel.__init__(self, normalize=normalize, em_optimisation=em_optimisation, weight_sp=weight_sp,ensemble_learning=ensemble_learning)

    def _deconvolution(self, mass_spec_mixture, cell_specific, weights=None,return_masks = False):
        mass_spec_mixture, cell_specific = mass_spec_mixture.copy(deep=True), cell_specific.copy(deep=True)

        ransac = linear_model.RANSACRegressor()
        ransac.fit(cell_specific, mass_spec_mixture)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        cell_abundance = pd.Series(data=ransac.estimator_.coef_, index=cell_specific.columns, name=mass_spec_mixture.name)
        cell_abundance[cell_abundance < 0] = 0

        if not return_masks :
            return cell_abundance
        return cell_abundance,inlier_mask,outlier_mask
