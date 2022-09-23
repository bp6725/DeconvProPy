import pandas as pd
import numpy as np
from functools import partial
import multiprocessing

from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
from functools import partial
from scipy.optimize import minimize
import scipy.optimize
from itertools import combinations
import matplotlib.pyplot as plt

from deconv_py.preprocess.base import BasePreprocess as PP_base
from deconv_py.preprocess.cell_specific import CellSpecific as PP_proteins

from deconv_py.infras.data_factory import DataFactory
from deconv_py.infras.data_loader import DataLoader

from deconv_py.models.base import Base as Models_base
from deconv_py.models.cell_proportions_models import CellProportions
from deconv_py.models.cell_specific_models import CellSpecificPerPermutation

from deconv_py.experiments.cell_specific import CellSpecificMetricsPlot


if __name__ == '__main__':
    data_factory = DataFactory()
    profile_data, profile_data_relvent_data = data_factory.build_cell_specific_profile()
    mixtures, mixtures_relvent_data = data_factory.build_mixture_data()

    cell_proportions_df = pd.DataFrame(index=['LFQ intensity NOT_CD4TCellTcm_01', 'LFQ intensity NOT_BCellmemory_01',
                                              'LFQ intensity NOT_Monocytesnonclassical_01'],
                                       columns=list(range(1, 9)),
                                       data=np.array(((100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 33), (25, 25, 50),
                                                      (25, 50, 25), (50, 25, 25), (47.5, 47.5, 5.0))).T)

    _profile_data, _mixtures = PP_base.return_mutual_proteins_by_index(profile_data, mixtures)
    A = _profile_data[profile_data_relvent_data]
    B = _mixtures[mixtures_relvent_data]

    X = cell_proportions_df

    B = B.rename({f: f.split('LFQ intensity ')[1] for f in B.columns}, axis=1)

    X = X.rename({f: f.split('LFQ intensity ')[1] for f in X.index}, axis=0)
    X = X.rename({f: 'mixture' + str(f) for f in X.columns}, axis=1)

    A = A.rename({f: f.split('LFQ intensity ')[1] for f in A.columns}, axis=1)

    cell_specific_per_permutation_nn_dfs,permuts = CellSpecificPerPermutation.nn_fit_as_df(X.T, B.T)

    res = CellSpecificMetricsPlot.std_over_permuts(cell_specific_per_permutation_nn_dfs, 'NOT_CD4TCellTcm')
    pass


















