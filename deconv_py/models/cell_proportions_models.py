import numpy as np
from scipy.optimize import least_squares
import scipy.optimize
import pandas as pd


class CellProportions():
    def __init__(self):
        pass

    @staticmethod
    def fit(cell_specific, mass_spec):
        """
        cell proportions using scipy.optimize.nnls
        :param cell_specific: [number_of_proteins,number_of_cells]
        :param mass_spec: [number_of_proteins,number_of_samples]
        :return: cell_abundance_over_samples:[number_of_proteins,number_of_samples]
        """
        cell_abundance_over_samples = np.zeros((cell_specific.shape[1], mass_spec.shape[1]))

        for sample_ind, sample in enumerate(mass_spec.T):
            cell_abundance = scipy.optimize.nnls(cell_specific, mass_spec[:, sample_ind])[0]
            cell_abundance = cell_abundance
            cell_abundance_over_samples[:, sample_ind] = cell_abundance
        cell_abundance_over_samples = cell_abundance_over_samples.T / cell_abundance_over_samples.sum(axis=0)[:,
                                                                      np.newaxis]
        cell_abundance_over_samples = cell_abundance_over_samples.round(2)
        return cell_abundance_over_samples

    @staticmethod
    def fit_as_df(cell_specific, mass_spec,**kwargs):
        """
        cell proportions using scipy.optimize.nnls
        :param cell_specific: [number_of_proteins,number_of_cells]
        :param mass_spec: [number_of_proteins,number_of_samples]
        :return: cell_abundance_over_samples:[number_of_proteins,number_of_samples]
        """
        cell_abundance_over_samples = []

        for sample in mass_spec:
            cell_abundance = scipy.optimize.nnls(cell_specific, mass_spec[sample])[0]
            cell_abundance_df = pd.DataFrame(data=cell_abundance, index=cell_specific.columns, columns=[sample])
            cell_abundance_over_samples.append(cell_abundance_df)

        cell_abundance_over_samples_df = pd.concat(cell_abundance_over_samples, axis=1)

        if "normalize" in kwargs.keys():
            if not kwargs["normalize"] :
                return cell_abundance_over_samples_df

        cell_abundance_over_samples_df = cell_abundance_over_samples_df / cell_abundance_over_samples_df.sum(
            axis=0)
        cell_abundance_over_samples_df = cell_abundance_over_samples_df.round(2)
        return cell_abundance_over_samples_df


