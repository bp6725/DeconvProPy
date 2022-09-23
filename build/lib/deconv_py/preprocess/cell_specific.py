import pandas as pd
from scipy.stats import entropy
from functools import reduce
from scipy.stats import entropy
from functools import reduce

class CellSpecific():
    def __init__(self,partial_pre_process_steps = [] ):
        self._partial_pre_process_steps = partial_pre_process_steps

    def transformer(self,A,B):
        _A,_B = A,B
        for partial_func in self._partial_pre_process_steps :
            _A,_B = partial_func(_A,_B)

        return _A,_B

    @staticmethod
    def pp_clean_irrelevant_proteins(A,B):
        _A,_B = A[A.columns[A.sum()>0]],B

        _a_relevant_proteins = A[A.sum(axis=1) > 0].index
        _b_relevant_proteins = B[B.sum(axis=1) > 0].index

        _relevant_proteins = _a_relevant_proteins.intersection(_b_relevant_proteins)
        return _A.loc[_relevant_proteins],_B.loc[_relevant_proteins]

    @staticmethod
    def pp_naive_discriminative_proteins(cell_specific, mass_spec, mixtures_trh=8, cells_trh=3):
        # keep proteins with less then mixtures_trh occurrences in B
        # keep proteins with less then cells_trh occurrences in A

        _cell_specific = cell_specific
        if cell_specific.shape[0] > cells_trh:
            _cell_specific = cell_specific.T

            # discriminative proteins
        mass_spec_dis_proteins = mass_spec[(mass_spec > 0).sum(axis=1) < mixtures_trh].index
        cell_specific_dis_proteins = _cell_specific[
            _cell_specific.columns[(_cell_specific > 0).sum() < cells_trh]].columns

        mutual_dis_proteins = (mass_spec_dis_proteins.intersection(cell_specific_dis_proteins))

        return cell_specific.loc[mutual_dis_proteins], mass_spec.loc[mutual_dis_proteins]

    @staticmethod
    def pp_margin_quantile(cell_specific, mass_spec, quantile=0.90):
        top_mergin = cell_specific.quantile(quantile)
        bottom_mergin = cell_specific.quantile(1 - quantile)

        index_in_condition = cell_specific[(cell_specific <= bottom_mergin)].dropna(how='all').index | cell_specific[
            (cell_specific >= top_mergin)].dropna(how='all').index
        topq_cell_specific = cell_specific.loc[index_in_condition]
        return topq_cell_specific, mass_spec.loc[topq_cell_specific.index]

    @staticmethod
    def pp_under_quantile(cell_specific, mass_spec, quantile=0.9):
        index_in_condition = cell_specific[(cell_specific <= cell_specific.quantile(quantile))].dropna(how='any').index
        topq_cell_specific = cell_specific.loc[index_in_condition]
        return topq_cell_specific, mass_spec.loc[topq_cell_specific.index]

    @staticmethod
    def pp_binary_occurrence(cell_specific, mass_spec, mixtures_trh=8, cells_trh=3):
        A,B= CellSpecific.pp_naive_discriminative_proteins(cell_specific, mass_spec, mixtures_trh, cells_trh)

        A[A> 0] = 1
        B[B> 0] = 1

        return A,B

    @staticmethod
    def pp_filter_by_protein_list(cell_specific, mass_spec,protein_list = []):
        idxs = pd.Index(protein_list)
        return cell_specific.loc[idxs], mass_spec.loc[idxs]

    @staticmethod
    def pp_entropy_based(_A, _B, n_genes_per_cell=20, gene_entropy_trh=0.001, with_norm=True,only_signature = False):
        _A_norm = _A.div(_A.sum(axis=1), axis=0)
        gene_entropy = _A_norm.apply(lambda gene_dis: entropy(gene_dis), axis=1)


        cell_to_list_of_max_genes = {}
        for protein, cell in _A.idxmax(axis=1).to_dict().items():
            if cell in cell_to_list_of_max_genes.keys():
                cell_to_list_of_max_genes[cell].append(protein)
            else:
                cell_to_list_of_max_genes[cell] = []

        list_of_genes_list = []

        for cell in _A.columns:
            _n_genes_per_cell = n_genes_per_cell
            cell_relvent_gene_entropy = gene_entropy[_A_norm[_A_norm[cell] > 0].index]
            #     cell_relvent_gene_entropy = cell_relvent_gene_entropy.loc[cell_to_list_of_max_genes[cell]].dropna()
            # takse genes with zero entropy and the largest values
            zero_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy[cell_relvent_gene_entropy == 0].index]
            best_zero_entropy_genes = zero_entropy_genes.nlargest(n_genes_per_cell).index

            best_genes_idx = best_zero_entropy_genes
            # how much genes we still need -
            _n_genes_per_cell = _n_genes_per_cell - best_zero_entropy_genes.shape[0]
            trh = gene_entropy_trh
            while best_genes_idx.shape[0] < _n_genes_per_cell:
                low_quantile_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy[
                    (cell_relvent_gene_entropy < cell_relvent_gene_entropy.quantile(trh)) & (
                                cell_relvent_gene_entropy > 0)].index]
                best_quantile_entropy_genes = low_quantile_entropy_genes.nlargest(_n_genes_per_cell).index
                best_genes_idx = best_genes_idx.union(best_quantile_entropy_genes)
                trh = trh + gene_entropy_trh
                if trh > 1:
                    print(cell)
                    break

            list_of_genes_list.append(best_genes_idx)

            list_of_genes_list.append(best_genes_idx)

        genes_list_idx = reduce(lambda x, y: x.union(y), list_of_genes_list)

        if only_signature :
            return genes_list_idx

        filt_A = _A.copy(deep=True).loc[genes_list_idx]
        filt_B = _B.copy(deep=True).loc[genes_list_idx]

        if not with_norm:
            return filt_A, filt_B
        else:
            norm_filt_A = filt_A.div(filt_A.max(axis=1), axis=0)
            norm_filt_B = filt_B.div(filt_A.max(axis=1), axis=0)
        return norm_filt_A, norm_filt_B

    @staticmethod
    def pp_entropy_largest_among_cells(_A, _B, n_genes_per_cell=20, gene_entropy_trh=0.001, with_norm=True,only_signature = False):
        _A_norm = _A.div(_A.sum(axis=1), axis=0)
        gene_entropy = _A_norm.apply(lambda gene_dis: entropy(gene_dis), axis=1)

        cell_to_list_of_max_genes = {}
        for protein, cell in _A.idxmax(axis=1).to_dict().items():
            if cell in cell_to_list_of_max_genes.keys():
                cell_to_list_of_max_genes[cell].append(protein)
            else:
                cell_to_list_of_max_genes[cell] = [protein]

        list_of_genes_list = []

        for cell in _A.columns:
            _n_genes_per_cell = n_genes_per_cell
            cell_relvent_gene_entropy = gene_entropy[_A_norm[_A_norm[cell] > 0].index]
            cell_relvent_gene_entropy = cell_relvent_gene_entropy.loc[cell_to_list_of_max_genes[cell]].dropna()

            # takse genes with zero entropy and the largest values
            zero_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy[cell_relvent_gene_entropy == 0].index]
            best_zero_entropy_genes = zero_entropy_genes.nlargest(_n_genes_per_cell).index

            best_genes_idx = best_zero_entropy_genes
            # how much genes we still need -
            _n_genes_per_cell = _n_genes_per_cell - best_zero_entropy_genes.shape[0]
            trh = gene_entropy_trh
            while best_genes_idx.shape[0] < _n_genes_per_cell:
                low_quantile_entropy_genes = _A[cell].loc[cell_relvent_gene_entropy[
                    (cell_relvent_gene_entropy < cell_relvent_gene_entropy.quantile(trh)) & (
                                cell_relvent_gene_entropy > 0)].index]
                best_quantile_entropy_genes = low_quantile_entropy_genes.nlargest(_n_genes_per_cell).index
                best_genes_idx = best_genes_idx.union(best_quantile_entropy_genes)
                trh = trh + gene_entropy_trh
                if trh > 1:
                    # print(cell)
                    break

            list_of_genes_list.append(best_genes_idx)

        genes_list_idx = reduce(lambda x, y: x.union(y), list_of_genes_list)

        if only_signature :
            return genes_list_idx

        filt_A = _A.copy(deep=True).loc[genes_list_idx]
        filt_B = _B.copy(deep=True).loc[genes_list_idx]

        if not with_norm:
            return filt_A, filt_B
        else:
            norm_filt_A = filt_A.div(filt_A.max(axis=1), axis=0)
            norm_filt_B = filt_B.div(filt_A.max(axis=1), axis=0)
        return norm_filt_A, norm_filt_B