import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

_profilte_to_cytof_cell_map = {'NOT_BCellmemory': 'B cells ',
                                  'NOT_BCellnaive': 'B cells ',
                                  'NOT_BCellplasma': 'B cells ',
                                  'NOT_CD4TCellmTregs': None,
                                  'NOT_CD4TCellnaive': 'Na?ve CD4 Tcell',
                                  'NOT_CD4TCellnTregs': None,
                                  # 'NOT_CD4TCellTcm': 'CD4+central memory Tcells',
                                  'NOT_CD4TCellTem': 'CD4+ effector memory T cells',
                                  'NOT_CD4TCellTemra': 'CD4+ effector memory T cells',
                                  'NOT_CD4TCellTh1': None,
                                  'NOT_CD4TCellTh17': None,
                                  'NOT_CD4TCellTh2': None,
                                  'NOT_CD8TCellnaive': 'Na?ve CD8 Tcell',
                                  'NOT_CD8TCellTcm': None,
                                  'NOT_CD8TCellTem': 'CD8+ effector memory T cells',
                                  'NOT_CD8TCellTemra': 'CD8+ effector memory T cells',
                                  'NOT_DendriticCD1c': 'Plasmacytoid dendritic cells',
                                  'NOT_DendriticCD304': 'Plasmacytoid dendritic cells',
                                  'NOT_Erythrocytes': None,
                                  'NOT_Monocytesclassical': 'Monocytes',
                                  'NOT_Monocytesintermediate': 'Monocytes',
                                  'NOT_Monocytesnonclassical': 'Monocytes',
                                  'NOT_NKCellsCD56bright': None,
                                  'NOT_NKCellsCD56dim': None,
                                  'NOT_Thrombocytes': None
    # ,
    #                               'NOT_Granulocyteseosinophils_01':None,
                                  # 'NOT_GranulocytesBasophil_01'
                               }

_cells_to_selected_cluster = {
    'B cells ': 'PB.1079966',
    'Na?ve CD4 Tcell': "PB.1079943",
    'CD4+central memory Tcells': "PB.1079837",
    'CD4+ effector memory T cells': "PB.1079987",
    'Na?ve CD8 Tcell': "PB.1079956",
    'CD8+ effector memory T cells': "PB.1079981",
    'Plasmacytoid dendritic cells': "PB.1079868",
    'Monocytes': "PB.1079983"}

class CytofCellCountInfra():
    def __init__(self, cluster_info_path= "../infras/cytof_data/raw_data/CyTOF.features.and.clusters.info.xlsx",
                 cytof_data_path="../infras/cytof_data/raw_data/filtered.esetALL.CyTOF.abundance.only.xlsx"):
        self.clusters_info_no_antygen = self.filter_out_antigens(pd.read_excel(cluster_info_path))

        cytof_data =  pd.read_excel(cytof_data_path).rename(columns = {'Unnamed: 0' : "mixtures"}).set_index("mixtures")
        self.cytof_data = cytof_data

    def filter_out_antigens(self,clusters_info):
        clusters_info = clusters_info.set_index("index")
        cytof_cluster_no_antigens = clusters_info[clusters_info.index.str.len() < 11].copy()
        cytof_cluster_no_antigens.loc[:, "featureID"] = cytof_cluster_no_antigens.loc[:,"featureID"].str.slice(3)
        cytof_cluster_no_antigens = cytof_cluster_no_antigens[
            cytof_cluster_no_antigens.featureID.apply(lambda x: x.isnumeric())]
        cytof_cluster_no_antigens.loc[:,"featureID"] = cytof_cluster_no_antigens.loc[:,"featureID"].apply(
            (lambda x: int(x) if x.isdigit() else x))

        return cytof_cluster_no_antigens

    def cytof_label_propagation(self,profile_df, profilte_to_cytof_cell_map = _profilte_to_cytof_cell_map,method = "dendrogram_label_propagation"):

        def find_label_of_neighbour_cell(cell, neighbour_number,distance_df):
            neighbour = distance_df.loc[cell].sort_values().index[neighbour_number]

            if neighbour in profilte_to_cytof_cell_map.keys():
                label = profilte_to_cytof_cell_map[neighbour]
                if label:
                    return label
            return find_label_of_neighbour_cell(cell, neighbour_number + 1,distance_df)

        if len([cm for cm in profile_df.columns if cm in profilte_to_cytof_cell_map]) == 0 :
            new_profilte_to_cytof_cell_map = {}

            for cell_from_profile in profile_df.columns :
                corr_cell_in_map = [c for c in profilte_to_cytof_cell_map if c in  cell_from_profile ]
                if len(corr_cell_in_map) == 0 :
                    new_profilte_to_cytof_cell_map[cell_from_profile] = None
                else :
                    new_profilte_to_cytof_cell_map[cell_from_profile] = profilte_to_cytof_cell_map[corr_cell_in_map[0]]

            profilte_to_cytof_cell_map = new_profilte_to_cytof_cell_map

        if method == "dendrogram_label_propagation" :
            new_profilte_to_cytof_cell_map = profilte_to_cytof_cell_map.copy()

            if len(profile_df.columns) > len(profile_df.index) :
                distance_df = pd.DataFrame(
                            squareform(pdist(profile_df)),
                    columns=profile_df.index,
                    index=profile_df.index)
            else :
                distance_df = pd.DataFrame(
                    squareform(pdist(profile_df.T)),
                    columns=profile_df.columns,
                    index=profile_df.columns)

            distance_df[distance_df == 0] = distance_df.max().max()
            for cell, label in new_profilte_to_cytof_cell_map.items():
                if label:
                    continue
                new_label = find_label_of_neighbour_cell(cell, 0,distance_df.copy())
                new_profilte_to_cytof_cell_map[cell] = new_label

        return new_profilte_to_cytof_cell_map

    def get_cytof_count_per_sample(self , cells_to_selected_cluster=_cells_to_selected_cluster, filter_by_version="V1"):
        relvent_cytof_data = self.cytof_data[[c for c in cells_to_selected_cluster.values()]]
        cells_over_samples_df = relvent_cytof_data.rename(
            columns={val: key for key, val in cells_to_selected_cluster.items()})
        self.samples_over_cells_df = cells_over_samples_df.T
        samples_over_cells_df = self.samples_over_cells_df[
            [col for col in self.samples_over_cells_df.columns if filter_by_version in col]]
        return samples_over_cells_df.copy(deep=True)/samples_over_cells_df.sum()

    def mass_to_cytof_count(self,mass_results_df, full_profilte_to_cytof_cell_map=None,avraged_sum_from_HA_cell_prop = False):

        def return_cytof_cell(mass_cell, full_profilte_to_cytof_cell_map):
            if mass_cell in full_profilte_to_cytof_cell_map.keys():
                return full_profilte_to_cytof_cell_map[mass_cell]
            return "Unknown"

        if full_profilte_to_cytof_cell_map is None :
            full_profilte_to_cytof_cell_map = _profilte_to_cytof_cell_map

        mass_results_df = mass_results_df.copy(deep=True)
        mass_results_df["cytof_cell"] = mass_results_df.index.map(
            lambda x: return_cytof_cell(x, full_profilte_to_cytof_cell_map))

        if not avraged_sum_from_HA_cell_prop :
            return mass_results_df.groupby("cytof_cell").sum()

        raise("not implmanted - benny remmber to add avrage sum using human cell atlas")

    def calculate_measurement(self,res_as_cytof , cytof_res,mass_results_df):
        return res_as_cytof.copy(deep=True).drop(columns=["Unknown"]).corr(cytof_res)

    def return_mass_and_cytof_not_none_cells_counts(self,mass_results_df,profilte_to_cytof_cell_map = None,filter_by_version="V1",is_mean_agg = False):
        cytof_not_none_count_df = self.get_cytof_count_per_sample(filter_by_version = filter_by_version)

        if self._is_cells_alredy_in_cytof_format(mass_results_df,_profilte_to_cytof_cell_map) :
            profilte_to_cytof_cell_map = {cell:cell for cell in mass_results_df.index}

        if profilte_to_cytof_cell_map is None :
            profilte_to_cytof_cell_map = _profilte_to_cytof_cell_map

        if len([cm for cm in mass_results_df.index if cm in profilte_to_cytof_cell_map]) == 0 :
            new_profilte_to_cytof_cell_map = {}

            for cell_from_profile in mass_results_df.index :
                corr_cell_in_map = [c for c in profilte_to_cytof_cell_map if c in cell_from_profile ]
                if len(corr_cell_in_map) == 0 :
                    new_profilte_to_cytof_cell_map[cell_from_profile] = None
                else :
                    new_profilte_to_cytof_cell_map[cell_from_profile] = profilte_to_cytof_cell_map[corr_cell_in_map[0]]

            profilte_to_cytof_cell_map = new_profilte_to_cytof_cell_map


        not_none_profile_cells = [profile_cell for profile_cell, cytof_cell in profilte_to_cytof_cell_map.items() if
                                  cytof_cell is not None]
        mass_results_not_none_df = mass_results_df.loc[not_none_profile_cells].copy(deep=True)

        mass_results_not_none_df["cytof_cell"] = mass_results_not_none_df.index.map(
            lambda x: profilte_to_cytof_cell_map[x])

        if not is_mean_agg :
            mass_results_not_none_df = mass_results_not_none_df.groupby("cytof_cell").sum()
        else :
            mass_results_not_none_df = mass_results_not_none_df.groupby("cytof_cell").mean()

        mutual_cells = mass_results_not_none_df.index.intersection(cytof_not_none_count_df.index)

        return mass_results_not_none_df.loc[mutual_cells],cytof_not_none_count_df.loc[mutual_cells]

    def _is_cells_alredy_in_cytof_format(self,mass_results_df,_cytof_cell_map):
        cytof_cells = [cc for cc in _cytof_cell_map.values()]
        for cell in  mass_results_df.index :
            if cell not in cytof_cells :
                return False
        return True