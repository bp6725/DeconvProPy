import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self,mass_spec_path=r'.\data\20150208_mixture_proteinGroups.xls',
                 protein_profile_path = r'.\data\20150718_Cerberus_proteinGroups.txt',as_csv = False,
                 profile_tag = "simple_ar"):

        self._profile_tag = profile_tag
        self._mass_spec_data = self.read_data_to_df(mass_spec_path,as_csv)
        self._protein_profile = self.read_data_to_df(protein_profile_path,as_csv)
        self._cell_proportions = pd.DataFrame(index = ['NOT_CD4TCellTcm_01','NOT_BCellmemory_01',
                                                      'NOT_Monocytesnonclassical_01'],
                                             columns = list(range(1,9)),
                                             data =np.array(((100,0,0),(0,100,0),(0,0,100),(33,33,33),(25,25,50)
                                                   ,(25,50,25),(50,25,25),(47.5,47.5,5.0))).T)

    def read_data_to_df(self, path,as_csv = False):
        try :
            if as_csv :
                return pd.read_csv(path)
            return pd.read_excel(path)
        except :
            return pd.read_csv(path,sep = "\t")

    def retrieve_all(self):
        return self._mass_spec_data, self._protein_profile, self._cell_proportions

    def get_mass_spec_data(self):
        return self._mass_spec_data.copy(deep=True)

    def get_protein_profile(self):
        return self._protein_profile.copy(deep=True)

    def get_cell_proportions(self):
        return self._cell_proportions.copy(deep=True)


if __name__ == '__main__':
    dl = DataLoader()