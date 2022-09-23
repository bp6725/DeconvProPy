from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
import pandas as pd
import scipy
from deconv_py.infras.cellMix.cellMix_coordinator import CellMixCoordinator
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator
import numpy as np
import random


class DeconvolutionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, normalize = True,em_optimisation = False,weight_sp=True,ensemble_learning = False):
        self.normalize = normalize
        self.em_optimisation = em_optimisation
        self.weight_sp = weight_sp
        self.ensemble_learning = ensemble_learning

    def fit(self, X=None, y=None):
        """
        cell proportions using scipy.optimize.nnls
        :param cell_specific: [number_of_proteins,number_of_cells]
        :param mass_spec: [number_of_proteins,number_of_samples]
        :return: cell_abundance_over_samples:[number_of_proteins,number_of_samples]
        """

        cell_specific, mass_spec = X[0], X[1]

        if (cell_specific is None) or (mass_spec is None):
            return None

        predicted_sp_genes = self._get_relevant_sp(cell_specific)
        weights = self._get_gene_weights(cell_specific, predicted_sp_genes)

        mass_spec, cell_specific = self._update_mixtures_by_weight(mass_spec,cell_specific,weights)
        #TODO: this is a way to impute the signature, its not dependnce on em_opt flag *
        if self.em_optimisation :
            cell_specific = self._impute_profile_data(predicted_sp_genes,cell_specific)

        if not self.ensemble_learning :
            cell_abundance_over_samples_df = self._run_deconvolution_over_samples(mass_spec, cell_specific, None,
                                                                              predicted_sp_genes)
        else :
            cell_abundance_over_samples_df = self._ensemble_weak_learners(mass_spec, cell_specific, None,predicted_sp_genes)

        if not self.normalize:
            return cell_abundance_over_samples_df

        cell_abundance_over_samples_df = (cell_abundance_over_samples_df / cell_abundance_over_samples_df.sum(
            axis=0)).round(2)

        return cell_abundance_over_samples_df

    def predict(self, X):
        return self.fit(X)

    def fit_predict(self, X=None, y=None):
        return self.predict("None")

    def transform(self, data):
        return data

    def _deconvolution(self, mass_spec_mixture, cell_specific, weights=None):
        raise NotImplementedError("_deconvolution")

# region private

    def _get_relevant_sp(self, cell_specific):
        ctnc = CtpNetCoordinator(path_of_cached_imputed_proteins =r"C:\Repos\deconv_py\deconv_py\cache\ctpnet_result_df.pkl")
        sp_profile = ctnc.return_imputed_proteins_for_cells(cell_specific.columns.to_list())

        all_genes_in_mixture = cell_specific.index.get_level_values(1).to_list()
        relevent_genes = [gene for gene in sp_profile.index if gene in all_genes_in_mixture]
        _sp_profile = sp_profile.loc[relevent_genes].copy(deep=True)

        gene_to_protein_map = {v:k for k,v in cell_specific.index}
        _sp_profile["protein"] = _sp_profile.index.map(gene_to_protein_map)
        sp_profile = _sp_profile.set_index("protein",append=True).swaplevel(1,0)

        return sp_profile

    def __calculate_weights_for_genes(self, all_mixture_genes, surface_proteins_genes, sum_ratio=1):
        surface_proteins_genes_in_mixtures = [gene for gene in surface_proteins_genes if gene in all_mixture_genes]

        common_genes_count = len(all_mixture_genes) - len(surface_proteins_genes_in_mixtures)
        special_gene_count = len(surface_proteins_genes_in_mixtures)
        normalization_factor = 2 * common_genes_count * special_gene_count

        common_weigth = special_gene_count / normalization_factor
        special_weight = common_genes_count / normalization_factor

        gene_to_weigth = {}
        for gene in all_mixture_genes:
            weight = special_weight if (gene in surface_proteins_genes_in_mixtures) else common_weigth
            gene_to_weigth[gene] = weight

        return gene_to_weigth

    def _get_gene_weights(self, cell_specific, predicted_sp_genes):
        if not self.weight_sp:
            return None
        gene_weights = self.__calculate_weights_for_genes(cell_specific.index.to_list(),
                                                          predicted_sp_genes.index.to_list())
        return gene_weights

    def _run_deconvolution_over_samples(self, mass_spec, cell_specific, weights, predicted_sp_genes):
        cell_abundance_over_samples = []

        for sample in mass_spec:
            if self.em_optimisation:
                cell_abundance = self.em_deconvolution(mass_spec[sample], cell_specific, predicted_sp_genes, weights)
            else:
                cell_abundance = self._deconvolution(mass_spec[sample], cell_specific, weights)

            cell_abundance_df = pd.DataFrame(data=cell_abundance, index=cell_specific.columns, columns=[sample])
            cell_abundance_over_samples.append(cell_abundance_df)

        return pd.concat(cell_abundance_over_samples, axis=1)

        # endregion

    def _update_mixtures_by_weight(self, mass_spec, cell_specific, weights = None):
        mass_spec, cell_specific = mass_spec.copy(deep=True), cell_specific.copy(deep=True)
        if weights is None :
            return  mass_spec, cell_specific

        mass_spec, cell_specific = mass_spec.copy(deep=True), cell_specific.copy(deep=True)

        tmp_mass_spec_as_list = []
        tmp_cell_specific_as_list = []

        for protein, gene in mass_spec.index:
            tmp_mass_spec_as_list.append(
                [protein, gene] + (mass_spec.loc[(protein, gene)] * np.sqrt(weights[(protein, gene)])).to_list())
            tmp_cell_specific_as_list.append(
                [protein, gene] + (cell_specific.loc[(protein, gene)] * np.sqrt(weights[(protein, gene)])).to_list())

        _mass_spec_mixture = pd.DataFrame(columns=["protein", "gene"] + mass_spec.columns.to_list(),
                                          data=tmp_mass_spec_as_list)
        mass_spec = _mass_spec_mixture.set_index(["protein", "gene"])
        _cell_specific = pd.DataFrame(columns=["protein", "gene"] + cell_specific.columns.to_list(),
                                      data=tmp_cell_specific_as_list)
        cell_specific = _cell_specific.set_index(["protein", "gene"])

        return mass_spec, cell_specific


    # region em method

    def _expectation_step(self,reconstructed_proportions,updated_mixture,cell_specific,
                          genes_weights,exogenous_expression,stratgy) :

        expected_surface_proteins = self._build_expected_sp(reconstructed_proportions, exogenous_expression,
                                                            updated_mixture.index)

        surface_proteins_of_mixture = updated_mixture.loc[
            updated_mixture.index.isin(expected_surface_proteins.index)]

        new_surface_protein_mixture_values = self.change_mixture_surface_proteins_based_on_expected_order(
            expected_surface_proteins, surface_proteins_of_mixture)

        updated_mixture = self.update_mixture_data(updated_mixture, new_surface_protein_mixture_values)

        return updated_mixture,cell_specific,genes_weights

    def _impute_profile_data(self,sp_profile, A_sig_with_sp):
        A_sig_with_sp = A_sig_with_sp.copy(deep=True)
        new_profile_list = []
        for cell in sp_profile.columns:
            org_data = A_sig_with_sp[cell]
            sp_data = sp_profile[cell].copy(deep=True)

            mutual_genes = org_data.loc[org_data.index.isin(sp_data.index.to_list())].index
            _sp_of_org_data = org_data.loc[mutual_genes]
            _sp_data_relevent = sp_data[mutual_genes].copy(deep=True)

            _sp_of_org_data_as_frame =_sp_of_org_data.to_frame()
            # _sp_of_org_data_as_frame.index = _sp_of_org_data_as_frame.index.set_names(["protein","genes"])
            # _sp_of_org_data_as_frame = _sp_of_org_data_as_frame.reset_index(level=0)
            _sp_data_relevent_as_frame = _sp_data_relevent.to_frame()

            merged = _sp_of_org_data_as_frame.merge(_sp_data_relevent_as_frame, left_index=True, right_index=True,
                                                      suffixes=('_org', '_impu'))
            sp_of_org_data, sp_data_relevent = merged[f"{_sp_data_relevent.name}_org"], merged[
                f"{_sp_data_relevent.name}_impu"]

            new_org_data = self.change_mixture_surface_proteins_based_on_expected_order(sp_data_relevent, sp_of_org_data)

            new_profile_list.append(self.update_mixture_data(org_data, new_org_data))

        impu_A_sig_with_sp = pd.concat(new_profile_list, axis=1)

        return impu_A_sig_with_sp

    def em_deconvolution(self, mass_spec_mixture, cell_specific, exogenous_expression,expectation_stratgy='', genes_weights=None):
        updated_mixture = mass_spec_mixture.copy(deep=True)

        reconstructed_proportions = self._deconvolution(updated_mixture, cell_specific, genes_weights)
        for i in range(4):
            updated_mixture,cell_specific,genes_weights = self._expectation_step(reconstructed_proportions,
                                                                                 updated_mixture,cell_specific,
                                                                                 genes_weights,exogenous_expression,
                                                                                 stratgy = expectation_stratgy)

            reconstructed_proportions = self._deconvolution(updated_mixture, cell_specific, genes_weights)

        return reconstructed_proportions

    def _build_expected_sp(self, reconstructed_proportions, sp_profile, genes_in_signature):
        expected_surface_proteins = sp_profile.dot(reconstructed_proportions.loc[sp_profile.columns])
        mutual_genes = expected_surface_proteins[expected_surface_proteins.index.isin(genes_in_signature)].index
        expected_surface_proteins = expected_surface_proteins._set_name(reconstructed_proportions.name)

        return expected_surface_proteins.loc[mutual_genes]

    def change_mixture_surface_proteins_based_on_expected_order(self, expected_surface_proteins,
                                                                mass_spec_mixture_surface_proteins):
        _expected_surface_proteins = expected_surface_proteins.sort_values()
        new_mass_spec_mixture_surface_proteins = pd.Series(index=_expected_surface_proteins.index,
                                                           name=_expected_surface_proteins.name,
                                                           data=np.sort(mass_spec_mixture_surface_proteins.values))
        return new_mass_spec_mixture_surface_proteins

    def update_mixture_data(self, mass_spec_mixture, new_mass_spec_mixture):
        mass_spec_mixture = mass_spec_mixture.copy(deep=True)

        for protein, new_val in new_mass_spec_mixture.items():
            mass_spec_mixture[protein] = new_val
        return mass_spec_mixture

    # endregion


    # region  weak learners

    def _ensemble_weak_learners(self,mass_spec, cell_specific, weights, predicted_sp_genes,number_of_learners=1000):
        list_of_results = []
        for idx_list in self._sample_proteins(number_of_learners,mass_spec.shape[0]):
            proteins = mass_spec.iloc[idx_list].index
            _mass_spec = mass_spec.loc[proteins].copy(deep=True)
            _cell_specific = cell_specific.loc[proteins].copy(deep=True)
            _cell_abundance_over_samples_df = self._run_deconvolution_over_samples(_mass_spec,_cell_specific,weights,predicted_sp_genes)
            list_of_results.append(_cell_abundance_over_samples_df)

        res = pd.concat(list_of_results, axis=1)
        reverse_res = res.T
        return reverse_res.groupby(reverse_res.index).apply(lambda x : x[(x>x.quantile(0.1)) & ((x<x.quantile(0.9)))].mean()).T

    def _sample_proteins(self,number_of_learners,n_of_features_in_data):
        idxs = range(n_of_features_in_data)
        n_features_options = np.random.randint(50, n_of_features_in_data, number_of_learners)
        for i,n_features in zip(range(number_of_learners), n_features_options):
            sampling = random.choices(idxs, k = n_features)
            yield sampling

    #endregion

