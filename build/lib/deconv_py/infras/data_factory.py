from deconv_py.infras.data_loader import DataLoader
# import deconv_py
import numpy as np
from itertools import chain
import os
import pandas as pd
from deconv_py.preprocess.base import BasePreprocess as PP_base
import random
import scipy
from scipy.stats import powerlaw
from scipy.stats import rankdata
from scipy.stats import norm


class DataFactory():
    # region CONSTS

    RELVENT_COLUMNS = ['Protein IDs', 'Majority protein IDs', 'Protein names', 'Gene names', 'Number of proteins',
                       'Unique + razor sequence coverage [%]', 'Unique sequence coverage [%]', 'Q-value',
                       'Intensity', 'iBAQ', 'Only identified by site', 'Reverse', 'Potential contaminant', 'id',
                       'Evidence IDs', 'MS/MS IDs', 'Best MS/MS']

    B_RELVENT_DATA = ['{} mixture1', '{} mixture2', '{} mixture3', '{} mixture4',
                      '{} mixture5', '{} mixture6', '{} mixture7', '{} mixture8']

    A_RELVENT_DATA = ['{} NOT_CD4TCellTcm_01', '{} NOT_BCellmemory_01',
                      '{} NOT_Monocytesnonclassical_01']

    IBD_A_RELVENT_DATA = ['{} NOT_BCellmemory_01', '{} NOT_BCellnaive_01', '{} NOT_BCellplasma_01',
                          '{} NOT_CD4TCellmTregs_01',
                          '{} NOT_CD4TCellnaive_01', '{} NOT_CD4TCellnTregs_01', '{} NOT_CD4TCellTcm_01',
                          '{} NOT_CD4TCellTem_01',
                          '{} NOT_CD4TCellTemra_01', '{} NOT_CD4TCellTh1_01', '{} NOT_CD4TCellTh17_01',
                          '{} NOT_CD4TCellTh2_01',
                          '{} NOT_CD8TCellnaive_01', '{} NOT_CD8TCellTcm_01', '{} NOT_CD8TCellTem_01',
                          '{} NOT_CD8TCellTemra_01',
                          '{} NOT_DendriticCD1c_01', '{} NOT_DendriticCD304_01', '{} NOT_Erythrocytes_01',
                          '{} NOT_GranulocytesBasophil_01',
                          '{} NOT_Granulocyteseosinophils_01', '{} NOT_Granulocytesneutrophils_01',
                          '{} NOT_Monocytesclassical_01', '{} NOT_Monocytesintermediate_01',
                          '{} NOT_Monocytesnonclassical_01', '{} NOT_NKCellsCD56bright_01', '{} NOT_NKCellsCD56dim_01',
                          '{} NOT_Thrombocytes_01']

    IBD_A_RELVENT_DATA_all_vs = ['{} NOT_BCellmemory_', '{} NOT_BCellnaive_', '{} NOT_BCellplasma_',
                           '{} NOT_CD4TCellmTregs_',
                           '{} NOT_CD4TCellnaive_', '{} NOT_CD4TCellnTregs_', '{} NOT_CD4TCellTcm_',
                           '{} NOT_CD4TCellTem_',
                           '{} NOT_CD4TCellTemra_', '{} NOT_CD4TCellTh1_', '{} NOT_CD4TCellTh17_',
                           '{} NOT_CD4TCellTh2_',
                           '{} NOT_CD8TCellnaive_', '{} NOT_CD8TCellTcm_', '{} NOT_CD8TCellTem_',
                           '{} NOT_CD8TCellTemra_',
                           '{} NOT_DendriticCD1c_', '{} NOT_DendriticCD304_', '{} NOT_Erythrocytes_',
                           '{} NOT_GranulocytesBasophil_',
                           '{} NOT_Granulocyteseosinophils_', '{} NOT_Granulocytesneutrophils_',
                           '{} NOT_Monocytesclassical_', '{} NOT_Monocytesintermediate_',
                           '{} NOT_Monocytesnonclassical_', '{} NOT_NKCellsCD56bright_', '{} NOT_NKCellsCD56dim_',
                           '{} NOT_Thrombocytes_']

    IBD_B_SIMPLE_DATA = ['24_v1', '24_v2', '24_v3', '26_v1', '26_v2',
       '26_v3', '27_v1', '27_v2', '27_v3', '28_v1', '28_v2', '28_v3', '29_v1',
       '29_v2', '29_v3', '30_v1', '30_v2', '30_v3', '31_v1', '31_v2', '31_v3',
       '32_v1', '32_v2', '32_v3', '44_v1', '44_v2', '44_v3', '46_v1',
       '46_v2_190716164330', '46_v3', '47_v1', '47_v2', '47_v3', '48_v1',
       '48_v2', '48_v3', '20_v1', '20_v2', '20_v3', '21_v1', '21_v2', '21_v3',
       '22_v1', '22_v2', '22_v3', '23_v1', '23_v2', '23_v3', '33_v1',
       '33_v2_190723014153', '33_v3', '35_v1', '35_v2', '35_v3', '36_v1',
       '36_v2', '36_v3', '37_v1', '37_v2', '37_v3', '38_v1', '38_v2', '38_v3',
       '39_v1', '39_v2', '39_v3_190722234036', '40_v1', '40_v3', '42_v1',
       '42_v2', '42_v3']

    GRANULOCYTES = ["GranulocytesBasophil","Erythrocytes","GranulocytesBasophil","Granulocyteseosinophils","Granulocytesneutrophils","Thrombocytes"]

    # endregion

    #region public functions

    def __init__(self,data_loader : DataLoader = None):
        self._data_loader = data_loader

        self._A_all_vs = None
        self._B_all_vs = None

    def build_mixture_data(self,intensity_type='LFQ intensity',auto_filter_by = True,
                relvent_data=None,relvent_columns = None,log2_transformation = False):
        raw_mixture_data = self._data_loader.get_mass_spec_data()

        relvent_columns = self.RELVENT_COLUMNS if (relvent_columns is None) else relvent_columns
        b_relvent_data = self._build_relvent_data(DataFactory.B_RELVENT_DATA,intensity_type) if (relvent_data is None) else relvent_data

        if auto_filter_by :
            if "Reverse" in raw_mixture_data.columns :
                raw_mixture_data = raw_mixture_data[
                    raw_mixture_data['Reverse'].isna() & raw_mixture_data['Potential contaminant'].isna() & raw_mixture_data[
                        'Only identified by site'].isna()]

        mixture_data = raw_mixture_data[list(set(relvent_columns + b_relvent_data))]

        for_calc_df = mixture_data.copy(deep=True)
        # for_calc_df=for_calc_df.rename({f: f.split('{0} '.format(intensity_type))[1] for f in b_relvent_data}, axis=1)

        if log2_transformation :
            numric_col = for_calc_df.select_dtypes(include=np.number).columns.tolist()
            for_calc_df[numric_col] = for_calc_df[numric_col].rpow(2)

        if (for_calc_df.mean().mean()<100) and (not log2_transformation):
            print("maybe the values are in log2 ? ")


        return for_calc_df, b_relvent_data

    def build_cell_specific_profile(self,intensity_type='LFQ intensity',auto_filter_by = True,
                relvent_data=None,relvent_columns = None):
        cell_spec_data = self._data_loader.get_protein_profile()

        relvent_columns = self.RELVENT_COLUMNS if (relvent_columns is None) else relvent_columns
        a_relvent_data = self._build_relvent_data(DataFactory.A_RELVENT_DATA, intensity_type) if (
                    relvent_data is None) else relvent_data

        exist_cols=cell_spec_data.columns
        exist_cols = [exist_col for exist_col in exist_cols if "NOT_" in exist_col]
        _a_relvent_data = list(chain(*[[exist_col for exist_col in exist_cols if relvent_column in exist_col] for relvent_column in
                     a_relvent_data]))

        if auto_filter_by:
            cell_spec_data = cell_spec_data[
                cell_spec_data['Reverse'].isna() & cell_spec_data['Potential contaminant'].isna() & cell_spec_data[
                    'Only identified by site'].isna()]

        ref_cell_spec = cell_spec_data[relvent_columns + _a_relvent_data]


        # ref_cell_spec = ref_cell_spec.rename({f: f.split('{0} '.format(intensity_type))[1] for f in a_relvent_data}, axis=1)

        return ref_cell_spec,_a_relvent_data

    #endregion

    #region specific profiles

    def load_simple_artificial_profile(self,intensity_type,index_func=lambda x:x.split(";")[0],sample_to_pick="01",
                                       log2_transformation = False,auto_filter_by = True,with_granulocytes = False):

        if self._data_loader is None :
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20150208_mixture_proteinGroups.xls",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",
                           profile_tag="simple_ar")

        if not self._data_loader._profile_tag == "simple_ar" :
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20150208_mixture_proteinGroups.xls",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",
                           profile_tag="simple_ar")

        if sample_to_pick == "all" :
            rd = ['Intensity NOT_CD4TCellTcm_01', 'Intensity NOT_BCellmemory_01',
                  'Intensity NOT_Monocytesnonclassical_01',
                  'Intensity NOT_CD4TCellTcm_02', 'Intensity NOT_BCellmemory_02',
                  'Intensity NOT_Monocytesnonclassical_02',
                  'Intensity NOT_CD4TCellTcm_03', 'Intensity NOT_BCellmemory_03',
                  'Intensity NOT_Monocytesnonclassical_03',
                  'Intensity NOT_CD4TCellTcm_04', 'Intensity NOT_BCellmemory_04',
                  'Intensity NOT_Monocytesnonclassical_04']
            profile_data, profile_data_relvent_data = self.build_cell_specific_profile(intensity_type=intensity_type,auto_filter_by = auto_filter_by,
                                                                                       relvent_data = rd)
        else :
            profile_data, profile_data_relvent_data = self.build_cell_specific_profile(intensity_type=intensity_type,
                                                                                   auto_filter_by = auto_filter_by)

        mixtures, mixtures_relvent_data = self.build_mixture_data(intensity_type=intensity_type,
                                                                  auto_filter_by = auto_filter_by,
                                                                  log2_transformation=log2_transformation)

        cell_proportions_df = pd.DataFrame(
            index=['NOT_CD4TCellTcm',
                   'NOT_BCellmemory',
                   'NOT_Monocytesnonclassical'],
            columns=[f"mixture{i}" for i in list(range(1, 9))],
            data=np.array(((100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 33), (25, 25, 50),
                           (25, 50, 25), (50, 25, 25),(47.5, 47.5, 5.0))).T)

        _profile_data, _mixtures = PP_base.return_mutual_proteins_by_index(profile_data, mixtures,
                                                                           index_func=index_func)

        A = _profile_data[profile_data_relvent_data]
        A = A.rename({f: f.split(f'{intensity_type} ')[1] for f in A.columns}, axis=1)

        B = _mixtures[mixtures_relvent_data + ["Gene names"]]
        B = B.rename({f: f.split(f'{intensity_type} ')[1] for f in B.columns if not (f== "Gene names") }, axis=1)

        A = A.merge(_profile_data["Gene names"].to_frame(), left_index=True,
                                  right_index=True).set_index(["Gene names"], append=True)
        B = B.set_index("Gene names", append=True)

        if not with_granulocytes :
            columns_without_gran = []
            for col in A.columns :
                if len([gran for gran in DataFactory.GRANULOCYTES if gran in col ]) == 0:
                    columns_without_gran.append(col)
            A=A[columns_without_gran]

        return A,B,cell_proportions_df

    def load_simple_IBD_profile(self,intensity_type,index_func=lambda x:x.split(";")[0],sample_to_pick="01",
                                   log2_transformation = True,auto_filter_by = True,with_granulocytes = False):
        if self._data_loader is None:
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20190801_filtered_imputed_data.csv",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",as_csv = True,
                           profile_tag="simple_IBD")

        if not self._data_loader._profile_tag == "simple_IBD":
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20190801_filtered_imputed_data.csv",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",as_csv = True,
                           profile_tag="simple_IBD")

        relvent_data = self._build_relvent_data(DataFactory.IBD_A_RELVENT_DATA, intensity_type)

        profile_data, profile_data_relvent_data= self.build_cell_specific_profile(intensity_type = intensity_type,
            auto_filter_by = True, relvent_data = relvent_data,relvent_columns = ["Majority protein IDs"])

        mixtures, mixtures_relvent_data = self.build_mixture_data( intensity_type = intensity_type,
                                                                   relvent_data = self.IBD_B_SIMPLE_DATA,
                                                                   relvent_columns = ["Majority protein IDs"],
                                                                   auto_filter_by = auto_filter_by,
                                                                   log2_transformation = log2_transformation)

        _profile_data, _mixtures = PP_base.return_mutual_proteins_by_index(profile_data, mixtures,
                                                                                     index_func = index_func)

        A = _profile_data[profile_data_relvent_data]
        A = A.rename({f: f.split(f'{intensity_type} ')[1] for f in A.columns}, axis=1)

        B = _mixtures[mixtures_relvent_data]

        if not with_granulocytes :
            columns_without_gran = []
            for col in A.columns :
                if len([gran for gran in DataFactory.GRANULOCYTES if gran in col ]) == 0:
                    columns_without_gran.append(col)
            A=A[columns_without_gran]


        return A, B

    def load_IBD_all_vs(self,intensity_type,index_func=lambda x:x.split(";")[0],log2_transformation = True,auto_filter_by = True,
                        with_granulocytes = False):
        if self._data_loader is None:
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20190801_filtered_imputed_data.csv",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",as_csv = True,
                           profile_tag="simple_IBD")

        if not self._data_loader._profile_tag == "simple_IBD":
            self._data_loader = \
                DataLoader(mass_spec_path=r"C:\Repos\deconv_py\deconv_py\data\20190801_filtered_imputed_data.csv",
                           protein_profile_path=r"C:\Repos\deconv_py\deconv_py\data\20150718_Cerberus_proteinGroups.txt",as_csv = True,
                           profile_tag="simple_IBD")

        relvent_data = self._build_relvent_data(DataFactory.IBD_A_RELVENT_DATA_all_vs, intensity_type)

        profile_data, profile_data_relvent_data= self.build_cell_specific_profile(
            auto_filter_by=True, relvent_data=relvent_data,
            relvent_columns=["Majority protein IDs", "Gene names"])
        mixtures, mixtures_relvent_data = self.build_mixture_data(relvent_data=self.IBD_B_SIMPLE_DATA,
                                                                  intensity_type = intensity_type,
                                                                  relvent_columns=["Majority protein IDs","Gene.names"],
                                                                  auto_filter_by=auto_filter_by,
                                                                  log2_transformation=log2_transformation)

        _profile_data_intensity, _mixtures = PP_base.return_mutual_proteins_by_index(profile_data, mixtures,
                                                                                     index_func=index_func)
        A_all_vs = _profile_data_intensity[profile_data_relvent_data].copy(deep=True)
        B_all_vs = _mixtures[mixtures_relvent_data+["Gene.names"]].copy(deep=True)

        A_all_vs = A_all_vs.merge(_profile_data_intensity["Gene names"].to_frame(), left_index=True,
                                  right_index=True).set_index(["Gene names"], append=True)
        B_all_vs = B_all_vs.rename(columns = {"Gene.names" : "Gene names"}).set_index("Gene names",append=True)

        if not with_granulocytes :
            columns_without_gran = []
            for col in A_all_vs.columns :
                if len([gran for gran in DataFactory.GRANULOCYTES if gran in col ]) == 0:
                    columns_without_gran.append(col)
            A_all_vs=A_all_vs[columns_without_gran]

        return A_all_vs,B_all_vs

    def load_IBD_vs_A_and_B_intensity(self,intensity_type,index_func=lambda x:x.split(";")[0],A_agg_method = "01",B_agg_method = "all",
                                   log2_transformation = True,auto_filter_by = True,with_granulocytes = False):
        A_all_vs, B_all_vs = self.load_IBD_all_vs(intensity_type,index_func,log2_transformation=log2_transformation,
                                                  auto_filter_by=auto_filter_by)

        if A_agg_method in ["01","02","03","04"] :
            A_intensity = A_all_vs.copy(deep=True)[[col for col in A_all_vs.columns if A_agg_method in col]].T

        if A_agg_method == "mean" :
            gene_to_cell_type = A_all_vs.copy(deep=True).T
            gene_to_cell_type["cell"] = gene_to_cell_type.index.map(
                lambda x: x.split('_0')[0])
            A_intensity = gene_to_cell_type.groupby("cell").mean()

        if A_agg_method == "max":
            gene_to_cell_type = A_all_vs.copy(deep=True).T
            gene_to_cell_type["cell"] = gene_to_cell_type.index.map(
                lambda x: x.split('_0')[0])
            A_intensity = gene_to_cell_type.groupby("cell").max()

        if B_agg_method == "all":
            B_intensity = B_all_vs.copy(deep=True).T

        if B_agg_method == "mean" :
            b_gene_to_cell_type = B_all_vs.copy(deep=True).T
            b_gene_to_cell_type["cell"] = b_gene_to_cell_type .index.map(lambda x: x.split('_v')[0])
            B_intensity = b_gene_to_cell_type.groupby("cell").mean()

        if B_agg_method == "max" :
            b_gene_to_cell_type = B_all_vs.copy(deep=True).T
            b_gene_to_cell_type["cell"] = b_gene_to_cell_type .index.map(lambda x: x.split('_v')[0])
            B_intensity = b_gene_to_cell_type.groupby("cell").max()

        if not with_granulocytes :
            columns_without_gran = []
            for col in A_all_vs.columns :
                if len([gran for gran in DataFactory.GRANULOCYTES if gran in col ]) == 0:
                    columns_without_gran.append(col)
            A_all_vs=A_all_vs[columns_without_gran]
            B_all_vs=B_all_vs[columns_without_gran]

            A_intensity = A_intensity.loc[columns_without_gran]

        return A_all_vs, B_all_vs ,A_intensity.T, B_intensity.T

    def load_cytof_data(self,intensity_type,index_func=lambda x:x.split(";")[0],log2_transformation = True,auto_filter_by = True):
        raise NotImplementedError()
        return None

    def build_simulated_data(self,intensity_type = "Intensity",index_func=lambda x:x.split(";")[0],log2_transformation= True,auto_filter_by = True,
                             number_of_mixtures = 50,percantage_to_zero =0.1,kurtosis_of_low_abundance = 0.8,saturation = 0.7,unquantified_cell_percentage = 30):
        if self._A_all_vs is None :
            A_all_vs, B_all_vs = self.load_IBD_all_vs(intensity_type,index_func=index_func,
                                                  log2_transformation = log2_transformation ,
                                                  auto_filter_by = auto_filter_by )
            self._A_all_vs = A_all_vs
            self._B_all_vs = B_all_vs
        else :
            A_all_vs = self._A_all_vs
            B_all_vs = self._B_all_vs

        simulated_profile = self._pick_random_value_from_intra_range(A_all_vs.copy())
        simulated_profile = simulated_profile.rename(
            {f: f.split(f'{intensity_type} NOT_')[1] for f in simulated_profile.columns}, axis=1)


        cell_ref_freq = pd.read_csv(r"C:\Repos\deconv_py\deconv_py\data\cell_references_frequency.csv")
        cell_to_freq_map = cell_ref_freq[["mass_name", "range_srt"]].set_index("mass_name").to_dict()["range_srt"]

        columns_for_x = [col.lstrip(" NOT").lstrip("_") for col in self._build_relvent_data(DataFactory.IBD_A_RELVENT_DATA, "")]
        simulated_X = self._build_random_proportions(columns_for_x, cell_to_freq_map,number_of_mixtures,unquantified_cell_percentage)

        simulated_mixtures = simulated_profile.dot(simulated_X)


        # add zeros :
        chance_to_be_zero = self._clac_chance_to_be_zero(simulated_mixtures, kurtosis_of_low_abundance, saturation)

        noise_location = chance_to_be_zero.applymap(lambda x: self._is_zero_noise(x,percantage_to_zero/ chance_to_be_zero.mean().mean() ))
        simulated_mixtures[noise_location] = 0

        return simulated_profile , simulated_X,simulated_mixtures


    #endregion

    #region private functions

    def _build_relvent_data(self,relvent_data_format,intensity_type) :
        return [data_col.format(intensity_type) for data_col in relvent_data_format]

    def _pick_random_value_from_intra_range(self,profile):
        cache_path = r"C:\Repos\deconv_py\deconv_py\cache\random_value_from_intra_range_df.pkl"
        if os.path.exists(cache_path ):
            random_value_from_intra_range_df = pd.read_pickle(cache_path )
            return random_value_from_intra_range_df

        gene_to_profile_data = profile.copy(deep=True).T
        gene_to_profile_data["cell"] = gene_to_profile_data.index.map(lambda x: x.split('_0')[0])

        min_gene_to_profile_data = gene_to_profile_data.groupby("cell").min().T
        max_gene_to_profile_data = gene_to_profile_data.groupby("cell").max().T
        number_of_zeros = gene_to_profile_data.groupby("cell").agg(lambda x: x.eq(0).sum()).T

        # random_profile = pd.DataFrame(index=min_gene_to_profile_data.index, columns=min_gene_to_profile_data.columns)
        random_profile = []
        for index, cell_data in min_gene_to_profile_data.iterrows():
            for cell in cell_data.index:
                min_val = min_gene_to_profile_data.loc[index, cell]
                max_val = max_gene_to_profile_data.loc[index, cell]
                n_zeros = number_of_zeros.loc[index, cell]

                if max_val == 0:
                    random_profile.append([index, cell, 0])
                    continue

                value = random.randrange(min_val, max_val)
                if self._is_zero_noise(n_zeros/4,1) :
                    value = 0

                random_profile.append([index, cell, value])

        random_profile_df = pd.DataFrame(data=random_profile)
        random_profile_df = random_profile_df.pivot(index=0, columns=1,values=2)

        random_profile_df.to_pickle(cache_path )
        return random_profile_df

    def _my_distribution(self,min_val, max_val, mean, std):
        scale = max_val - min_val
        location = min_val
        # Mean and standard deviation of the unscaled beta distribution
        unscaled_mean = (mean - min_val) / scale
        unscaled_var = (std / scale) ** 2
        # Computation of alpha and beta can be derived from mean and variance formulas
        t = unscaled_mean / (1 - unscaled_mean)
        beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
        alpha = beta * t
        # Not all parameters may produce a valid distribution
        if alpha <= 0 or beta <= 0:
            raise ValueError('Cannot create distribution for the given parameters.')
        # Make scaled beta distribution with computed parameters
        return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

    def _build_random_proportions(self,columns , cell_to_freq_map, number_of_mixtures,unquantified_cell_percentage = 30):
        all_cells = columns
        cells_dist_params = {}
        for cell, params in cell_to_freq_map.items():
            params_list = [float(params.split("min:")[1][:4]), float(params.split("avg:")[1][:4]),
                           float(params.split("max:")[1][:4])]
            cells_dist_params[cell] = params_list
            all_cells.remove(cell)

        noise_ratio = np.round(unquantified_cell_percentage / len(all_cells), 2)
        for cell in all_cells:
            cells_dist_params[cell] = [0, noise_ratio, 2 * noise_ratio]

        X = pd.DataFrame(columns=[str(c) for c in range(number_of_mixtures)])
        for cell, params in cells_dist_params.items():
            dist = self._my_distribution(params[0], params[2], params[1], params[1] / 4)
            cell_name = cell.split("_0")[0]
            X.loc[cell_name] = dist.rvs(number_of_mixtures)
        return X / X.sum()

    def _is_zero_noise(self,chance, percantage_to_zero):
        if np.random.random() < chance:
            return np.random.random() < percantage_to_zero
        return False

    def _return_powelaw_per_gene(self,seq_data, kurtosis_of_low_abundance, saturation):
        pl = powerlaw(kurtosis_of_low_abundance, scale=saturation)

        percantile = (rankdata(seq_data, "average") / len(seq_data))
        return pl.cdf(percantile)

    def _clac_chance_to_be_zero(self,data, kurtosis_of_low_abundance=0.99, saturation=0.7):
        '''
        chance to be zero : norm_cdf(-(mean)/(powerlaw(mean)))
        '''

        chances = {}
        for mixture in data.columns:
            seq_data = data[mixture]
            stds = self._return_powelaw_per_gene(seq_data, kurtosis_of_low_abundance, saturation)
            norm_for_cdf = -1.0 / stds
            chance_to_zero = norm.cdf(norm_for_cdf)

            chance_to_zero /= chance_to_zero.max()
            chance_to_zero = 1 - chance_to_zero
            chance_to_zero_series = pd.Series(data=chance_to_zero, index=seq_data.index, name=seq_data.name)

            chances[mixture] = chance_to_zero_series

        return pd.DataFrame(chances)

    #endregion

if __name__ == '__main__':
    df = DataFactory()

    A, B,C = df.build_simulated_data()
    print(B.describe())