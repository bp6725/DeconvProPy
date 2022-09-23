import os
import pandas as pd
import numpy as np

from deconv_py.preprocess.base import BasePreprocess as PP_base
from deconv_py.preprocess.cell_specific import CellSpecific as PP_proteins
from deconv_py.infras.data_factory import DataFactory
from deconv_py.infras.data_loader import DataLoader
from deconv_py.models.cell_proportions_models import CellProportions

def main() :
    data_loader = DataLoader(mass_spec_path=os.path.abspath('../data/20150208_mixture_proteinGroups.xls'),
                             protein_profile_path=os.path.abspath('../data/20150718_Cerberus_proteinGroups.txt'))

    data_factory = DataFactory(data_loader)
    profile_data, profile_data_relvent_data = data_factory.build_cell_specific_profile(intensity_type='iBAQ')
    mixtures, mixtures_relvent_data = data_factory.build_mixture_data(intensity_type='iBAQ')

    cell_proportions_df = pd.DataFrame(
        index=['iBAQ NOT_CD4TCellTcm_01', 'iBAQ NOT_BCellmemory_01', 'iBAQ NOT_Monocytesnonclassical_01'],
        columns=list(range(1, 9)),
        data=np.array(((100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 33), (25, 25, 50), (25, 50, 25), (50, 25, 25),
                       (47.5, 47.5, 5.0))).T)

    index_func = lambda x: x.split(';')[0]
    _profile_data, _mixtures = PP_base.return_mutual_proteins_by_index(profile_data, mixtures, index_func=index_func)
    A = _profile_data[profile_data_relvent_data]
    B = _mixtures[mixtures_relvent_data]

    X = cell_proportions_df

    B = B.rename({f: f.split('iBAQ ')[1] for f in B.columns}, axis=1)

    X = X.rename({f: f.split('iBAQ ')[1] for f in X.index}, axis=0)
    X = X.rename({f: 'mixture' + str(f) for f in X.columns}, axis=1)

    A = A.rename({f: f.split('iBAQ ')[1] for f in A.columns}, axis=1)

    _A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
    _A, _B = PP_proteins.pp_naive_discriminative_proteins(_A, _B)

    cell_abundance_over_samples = CellProportions.fit(_A, _B.values)
    print(cell_abundance_over_samples)
    print(X)


if __name__ == '__main__':
    # main()
    print('hello')

