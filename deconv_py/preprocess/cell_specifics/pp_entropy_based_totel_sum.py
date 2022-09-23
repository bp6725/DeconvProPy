from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import entropy
from functools import reduce
import pandas as pd
from sklearn.base import BaseEstimator
from deconv_py.infras.data_factory import DataFactory
import numpy as np
import infras.cache_decorator as cache_decorator


class PpEntropyBasedTotelSum(TransformerMixin,BaseEstimator):
    def __init__(self,  totel_sum_percentage=0.01,number_of_bins = 0 , with_norm=True,low_intra_var_trh = None,only_largest = False):
        self.totel_sum_percentage = totel_sum_percentage
        self.with_norm= with_norm
        self.low_intra_var_trh = low_intra_var_trh
        self.only_largest=only_largest
        self.number_of_bins = number_of_bins

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        _A,_B = data[0],data[1]

        _A_norm = _A.div(_A.sum(axis=1), axis=0)

        if self.number_of_bins == 0  :
            gene_entropy = _A_norm.apply(lambda gene_dis: entropy(gene_dis), axis=1)
        else :
            gene_entropy = _A_norm.apply(lambda x:entropy(np.histogram(x,bins=self.number_of_bins)[0]),axis=1)

        if self.only_largest :
            cell_to_list_of_max_genes = {}
            for protein, cell in _A.idxmax(axis=1).to_dict().items():
                if cell in cell_to_list_of_max_genes.keys():
                    cell_to_list_of_max_genes[cell].append(protein)
                else:
                    cell_to_list_of_max_genes[cell] = [protein]

            #filter by intra var
            if _A.deconv.intra_variance is not None :
                method, intra_var = _A.deconv.intra_variance.popitem()
                trh = _A.deconv.intra_variance_trh[method]
                relvent_cells = {}
                for cell, proteins in cell_to_list_of_max_genes.items():
                    relvent_cells[cell] = []
                    for protein in proteins :
                        _val = intra_var.loc[protein,cell]
                        if _val < trh :
                            if cell in relvent_cells.keys():
                                relvent_cells[cell].append(protein)
                            else :
                                relvent_cells[cell] = [protein]
            else :
                relvent_cells = cell_to_list_of_max_genes

        else :
            relvent_cells = None
            if _A.deconv.intra_variance is not None:
                method, intra_var = _A.deconv.intra_variance.popitem()
                trh = _A.deconv.intra_variance_trh[method]

                relvent_cells = {}
                for protein, cell in _A.idxmax(axis=1).to_dict().items():
                    _val = intra_var.loc[protein, cell]
                    if _val < trh:
                        if cell in relvent_cells.keys():
                            relvent_cells[cell].append(protein)
                        else:
                            relvent_cells[cell] = [protein]

        sum_trh = _A.sum().mean()*self.totel_sum_percentage

        list_of_genes_list = []
        for cell in _A.columns:
            cell_gene_entropy = gene_entropy[_A_norm[_A_norm[cell] > 0].index]
            if relvent_cells is not None :
                if cell not in relvent_cells.keys() :
                    cell_relvent_gene_entropy = cell_gene_entropy
                else :
                    cell_relvent_gene_entropy = cell_gene_entropy.loc[relvent_cells[cell]].dropna()

                if cell_relvent_gene_entropy.shape[0] == 0:
                    cell_relvent_gene_entropy = cell_gene_entropy
            else :
                cell_relvent_gene_entropy = cell_gene_entropy


            cell_series = _A[_A[cell] > 0][cell]
            cell_with_entropy = pd.concat([cell_series, cell_relvent_gene_entropy], axis=1).dropna(axis=0)
            #we sort by entropy and then cumsum : so the start is the lowest entropy
            cum_sumed_values = cell_with_entropy.sort_values(by=0)[cell].cumsum()

            list_of_genes_list.append(self._get_genes_until_sumed_trh(cum_sumed_values,sum_trh))

        genes_list_idx = reduce(lambda x, y: x.union(y), list_of_genes_list)

        if _A.deconv.must_contain_genes is not None :
            genes_list_idx = genes_list_idx.union(_A.deconv.must_contain_genes).drop_duplicates()

        filt_A = _A.copy(deep=True).loc[genes_list_idx]
        filt_B = _B.copy(deep=True).loc[genes_list_idx]

        if not self.with_norm:
            try:
                filt_A.deconv.transfer_all_relevant_properties(_A)
                return [filt_A, filt_B]
            except:
                return [filt_A, filt_B]
        else:
            norm_filt_A = filt_A.div(filt_A.max(axis=1), axis=0)
            norm_filt_B = filt_B.div(filt_A.max(axis=1), axis=0)

        try:
            norm_filt_A.deconv.transfer_all_relevant_properties(_A)
            return [norm_filt_A, norm_filt_B]
        except:
            return [norm_filt_A, norm_filt_B]

    def fit(self, *_):
        return self

    def _get_genes_until_sumed_trh(self,cum_sumed_values,trh):
        ll = []
        for gene in cum_sumed_values.index:
            ll.append(gene)
            if cum_sumed_values[gene] > trh:
                break
        return pd.Index(ll)

if __name__ == '__main__':
    df = DataFactory()
    A,X,B = df.build_simulated_data()

    pp = PpEntropyBasedTotelSum()
    res = pp.transform([A,B])
    print("finish")
