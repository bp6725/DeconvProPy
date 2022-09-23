from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import entropy
from functools import reduce
import pandas as pd
import numpy as np
import infras.cache_decorator as cache_decorator

class PpEntropyBased(TransformerMixin,BaseEstimator):
    def __init__(self,  n_genes_per_cell=20,number_of_bins = 0, gene_entropy_trh=1, with_norm=True,only_signature = False):
        self.n_genes_per_cell = n_genes_per_cell
        self.gene_entropy_trh = gene_entropy_trh
        self.with_norm= with_norm
        self.number_of_bins = number_of_bins

    @cache_decorator.tree_cache_deconv_pipe
    def transform(self, data, *_):
        if data[0].empty :
            return None,None

        _A,_B = data[0],data[1]
        _A_norm = _A.div(_A.sum(axis=1), axis=0)

        if self.number_of_bins == 0  :
            gene_entropy = _A_norm.apply(lambda gene_dis: entropy(gene_dis), axis=1)
        else :
            gene_entropy = _A_norm.apply(lambda x:entropy(np.histogram(x,bins=self.number_of_bins)[0]),axis=1)


        # filter by intra var
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
                    else :
                        relvent_cells[cell] = [protein]

        list_of_genes_list = []

        for cell in _A.columns:
            _n_genes_per_cell = self.n_genes_per_cell
            cell_relvent_gene_entropy = gene_entropy[_A_norm[_A_norm[cell] > 0].index].copy(deep=True)
            if relvent_cells is not None :
                if cell not in relvent_cells.keys() :
                    continue
                cell_relvent_gene_entropy = cell_relvent_gene_entropy.loc[relvent_cells[cell]].dropna()

            if cell_relvent_gene_entropy.shape[0] < _n_genes_per_cell :
                best_genes_idx = cell_relvent_gene_entropy.index
            else :
                # takse genes with zero entropy and the largest values
                zero_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy[cell_relvent_gene_entropy == 0].index]
                best_zero_entropy_genes = zero_entropy_genes.nlargest(self.n_genes_per_cell).index

                best_genes_idx = best_zero_entropy_genes
                # how much genes we still need -
                _n_genes_per_cell = _n_genes_per_cell - best_zero_entropy_genes.shape[0]

                best_quantile_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy.nsmallest(_n_genes_per_cell*self.gene_entropy_trh).index].\
                    nlargest(_n_genes_per_cell).index

                best_genes_idx = best_genes_idx.union(best_quantile_entropy_genes)

            list_of_genes_list.append(best_genes_idx)

            list_of_genes_list.append(best_genes_idx)

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