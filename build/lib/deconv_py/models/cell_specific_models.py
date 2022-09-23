import numpy as np
import pandas as pd
from itertools import combinations
from deconv_py.models.base import Base
from scipy.optimize import nnls


class CellSpecificPerPermutation():
    def __init__(self):
        pass

    @staticmethod
    def fit(cell_proportions, mass_spec,
            opt_func=lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]):
        """
        :param cell_proportions:[number_of_samples,number_of_cells]
        :param mass_spec:[number_of_samples,number_of_proteins]
        :return: experiment_results:[number_of_cells,number_of_proteins,binomial_coefficients(number_of_cells/number_of_samples)]
        """

        cell_proportions = cell_proportions.values
        mass_spec = mass_spec.values

        number_of_cells = cell_proportions.shape[1]
        number_of_pbmcs, number_of_proteins = mass_spec.shape

        all_pbmc_permutations = list(combinations(range(number_of_pbmcs), number_of_cells))
        experiment_results = np.zeros((number_of_cells, number_of_proteins, len(all_pbmc_permutations)))

        for exp_ind, permut in enumerate(all_pbmc_permutations):
            experiment_cell_proportions = cell_proportions[permut, :]
            experiment_B = mass_spec[permut, :]
            experiment_cell_specific = opt_func(experiment_cell_proportions, experiment_B)
            experiment_results[:, :, exp_ind] = experiment_cell_specific

        return experiment_results

    @staticmethod
    def fit_as_df(cell_proportions, mass_spec,
                  opt_func=lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]):
        """
        :param cell_proportions:[number_of_samples,number_of_cells]
        :param mass_spec:[number_of_samples,number_of_proteins]
        :return: experiment_results:[number_of_cells,number_of_proteins,binomial_coefficients(number_of_cells/number_of_samples)]
        """
        cell_proportions = cell_proportions
        mass_spec = mass_spec

        number_of_cells = cell_proportions.shape[1]
        number_of_pbmcs, number_of_proteins = mass_spec.shape

        all_pbmc_permutations = list(combinations(range(number_of_pbmcs), number_of_cells))
        experiment_results = []
        permuts = []
        for _, permut in enumerate(all_pbmc_permutations):
            experiment_cell_proportions = cell_proportions.iloc[list(permut)]
            experiment_B = mass_spec.iloc[list(permut)]
            experiment_cell_specific = opt_func(experiment_cell_proportions, experiment_B)
            experiment_cell_specific_df = pd.DataFrame(index=experiment_cell_proportions.columns,
                                                       columns=experiment_B.columns,
                                                       data=experiment_cell_specific)

            experiment_results.append(experiment_cell_specific_df)
            permuts.append(permut)

        return experiment_results, permuts


    @staticmethod
    def nn_fit(cell_proportions, mass_spec):
        return CellSpecificPerPermutation.fit(cell_proportions, mass_spec,Base.nn_opt_func)

    @staticmethod
    def nn_fit_as_df(cell_proportions, mass_spec):
        return CellSpecificPerPermutation.fit_as_df(cell_proportions, mass_spec,Base.nn_opt_func)

