from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import entropy
from functools import reduce,lru_cache
import pandas as pd
import os
from operator import or_ as union


from deconv_py.infras.global_utils import GlobalUtils
import infras.cache_decorator as cache_decorator

class PpDepDeBased(TransformerMixin,BaseEstimator):
    def __init__(self,  n_of_genes=20,intansity_type = "Intensity",is_agg_cells = False):
        self.n_genes_per_cell = n_of_genes
        self.intansity_type = intansity_type
        self.is_agg_cells = is_agg_cells

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        if data[0].empty :
            return None,None

        if not (data[0].deconv.is_agg_cells_profile == self.is_agg_cells) :
            return None, None

        _A,_B = data[0],data[1]
        per_cell_de = self.get_per_cell_results(self.intansity_type,self.is_agg_cells)

        list_of_cells = _A.columns.map(lambda x:x.replace(" ",".").replace("+",".").replace("?","."))
        mapping = GlobalUtils.get_corospanding_cell_map_from_lists(list_of_cells, per_cell_de.keys())
        list_of_cells = [mapping[c] for c in list_of_cells]
        A_rank = _A.rank(axis=1).copy(deep=True).rename(columns = {cell:mapping[cell.replace(" ",".").replace("+",".").replace("?",".")] for cell in _A.columns})

        possible_genes = _A.index
        all_genes_from_de = reduce(union, (df.index for df in per_cell_de.values()))
        mapp_genes_idx = self._build_mapping_between_genes_idxs(all_genes_from_de, possible_genes)

        genes = self.get_sig_genes(per_cell_de,A_rank, list_of_cells,possible_genes,mapp_genes_idx)

        if _A.deconv.must_contain_genes is not None :
            genes = genes.union(_A.deconv.must_contain_genes).drop_duplicates()

        A_res,B_res = _A.loc[genes], _B.loc[genes]
        try:
            A_res.deconv.transfer_all_relevant_properties(_A)
            return [A_res, B_res]
        except:
            return [A_res, B_res]

    def fit(self, *_):
        return self

    @lru_cache(2)
    def get_per_cell_results(self,intansity_type, is_sumed_cell_types):
        if is_sumed_cell_types :
            file_name = f"AGG_{intansity_type}_DEP_DE.csv"
        else :
            file_name = f"{intansity_type}_DEP_DE.csv"

        dic_path = r"C:\Repos\deconv_py\deconv_py\data\de_methods_results"
        de_df = pd.read_csv(os.path.join(dic_path,file_name), sep=";")
        de_df = de_df.set_index(["name", "ID"])

        de_df = de_df[list(filter(lambda col: "_p.val" in col, de_df.columns))]
        de_df = de_df.applymap(lambda x: float(x.replace(",", ".")))

        per_cell_results_dicts = {}
        for col in de_df.head().columns:
            first, second = col.split("_p.val")[0].split("_vs_")

            if first in per_cell_results_dicts.keys():
                per_cell_results_dicts[first][second] = de_df[col]
            else:
                per_cell_results_dicts[first] = {second: de_df[col]}

            second, first = col.split("_p.val")[0].split("_vs_")

            if first in per_cell_results_dicts.keys():
                per_cell_results_dicts[first][second] = de_df[col]
            else:
                per_cell_results_dicts[first] = {second: de_df[col]}

            results = {}
        for cell in per_cell_results_dicts.keys():
            per_cel = per_cell_results_dicts[cell]
            results[cell] = pd.DataFrame(per_cel)

        return results

    def get_sig_genes(self,per_cell_de,A_rank, list_of_cells,possible_genes,mapp_genes_idx, n_of_genes=15):
        rank_trh = A_rank.shape[1]-3
        all_genes_list = []
        #we whernt able to map all genes,and we kept only possible genes
        existing_dep_genes =pd.Index(list(mapp_genes_idx.keys()))
        for cell in list_of_cells:
            flag = False
            de_df_per_cell = per_cell_de[cell]

            mutual_genes = existing_dep_genes.intersection(de_df_per_cell.index)
            de_df_per_cell = de_df_per_cell.reindex(mutual_genes)
            de_df_per_cell.index = de_df_per_cell.index.map(mapp_genes_idx)
            de_df_per_cell = de_df_per_cell.loc[~de_df_per_cell.index.duplicated(keep='first')]

            # filter cells not in the list
            de_df_per_cell = de_df_per_cell[[c for c in list_of_cells if c != cell]]

            #keep only reletive high genes for the specific cell
            high_ranked_genes =  A_rank[cell][A_rank[cell] > rank_trh].index
            de_df_per_cell = de_df_per_cell.loc[high_ranked_genes]

            for pval_trh in np.linspace(0.001, 0.1, 10):
                _de_df = de_df_per_cell[de_df_per_cell < pval_trh].dropna(how="any")
                if _de_df.shape[0] >= n_of_genes:
                    if _de_df.shape[0] > 2 * n_of_genes:
                        all_genes_list.append(_de_df.mean(axis=1).nsmallest(2 * n_of_genes).index)
                    else:
                        all_genes_list.append(_de_df.index)

                    flag = True
                    break
            if not flag:
                if _de_df.empty:
                    all_genes_list.append(de_df_per_cell.mean(axis=1).nsmallest(int(n_of_genes / 2)).index)
                else:
                    all_genes_list.append(_de_df.index)

        #         print(f"cell : {cell}, pval :{pval_trh},n : {len(all_genes_list[-1])}")
        all_genes = reduce(lambda x, y: x.union(y), all_genes_list)
        return all_genes

    def _build_mapping_between_genes_idxs(self,genes, a_genes):
        poss = {}
        for gene in genes:
            for a_gene in a_genes:
                if (a_gene[1] is np.nan) or (gene[1] is np.nan):
                    continue
                if (a_gene[0] in gene[0]) or (gene[0] in a_gene[0]) or (gene[0] in a_gene[0]) or (a_gene[0] in gene[0]):
                    if gene not in poss.keys():
                        poss[gene] = []
                    poss[gene].append(a_gene)

        genes_to_change = []
        for (k, vs) in filter(lambda x: len(x[1]) > 1, poss.items()):
            for v in vs:
                if (v[0] == k[0]) and (v[1] == k[1]):
                    genes_to_change.append((k, v))

        for g in genes_to_change:
            poss[g] = [g]

        final = {o: n[0] for o, n in poss.items()}
        return final