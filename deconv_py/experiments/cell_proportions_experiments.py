import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append('../infras/cellMix/')
sys.path.append('../infras/cytof_data/')
import os
# print(os.listdir('../infras/cytof_data/'))
from cellMix_coordinator import CellMixCoordinator
from cytof_cell_count_infra import CytofCellCountInfra
from deconv_py.models.cell_proportions_models import CellProportions
from deconv_py.preprocess.base import BasePreprocess as PP_base
from deconv_py.preprocess.cell_specific import CellSpecific as PP_proteins

class CellProportionsExperiments():

    def __init__(self):
        self.cci = CytofCellCountInfra()
        self.cmc = CellMixCoordinator()

    def _calc_and_display_with_cellmix(self,_a, _b, X, with_cellMix=False, as_heatmap=False, with_cytof=False,**kwargs):
        def get_mass_to_cytof_count(cell_abundance_over_samples, full_profilte_to_cytof_cell_map):
            return self.cci.mass_to_cytof_count(cell_abundance_over_samples, full_profilte_to_cytof_cell_map)

        def get_results(_a, _b, X, full_profilte_to_cytof_cell_map,**kwargs):
            cell_abundance_over_samples = CellProportions.fit_as_df(_a, _b,**kwargs)
            mass_cytof_results_df = None
            if full_profilte_to_cytof_cell_map is not None:
                mass_cytof_results_df = get_mass_to_cytof_count(cell_abundance_over_samples,
                                                                full_profilte_to_cytof_cell_map)

            return cell_abundance_over_samples, mass_cytof_results_df

        def get_cell_mix_results(_a, _b, X, with_cellMix, full_profilte_to_cytof_cell_map):
            if not with_cellMix:
                return None, None
            try :
                cellMax_cell_abundance_over_samples = self.cmc.cell_prop_with_bash(_b, _a).rename({"Unnamed: 0": "cells"},
                                                                                             axis=1).set_index("cells")
            except :
                print("cant use cellmix")
                return None,None

            if with_cytof:
                mass_cytof_results_df = get_mass_to_cytof_count(cellMax_cell_abundance_over_samples,
                                                                full_profilte_to_cytof_cell_map)
                return cellMax_cell_abundance_over_samples, mass_cytof_results_df
            return cellMax_cell_abundance_over_samples,None

        def display_result(cell_abundance, res_as_cytof, as_heatmap, added_title=""):
            if cell_abundance is None:
                return None
            if res_as_cytof is not None:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                sns.heatmap(res_as_cytof.sort_index(), ax=axs[1])
                axs[1].set_title("mass as cytof results" + added_title)

            if as_heatmap:
                if res_as_cytof is None:
                    fig, axs = plt.subplots(1, 1)
                sns.heatmap(cell_abundance, ax=axs[0])
                axs[0].set_title("cell abundance" + added_title)
                plt.subplots_adjust(wspace=1)
                plt.show()
            else:
                print(cell_abundance.style.set_caption("cell abundance" + added_title))

        if _a.empty:
            raise Exception("A is empty")
        if _b.empty:
            raise Exception("B is empty")

        full_profile_to_cytof_cell_map, sample_over_cytof_count = None, None
        if with_cytof:
            full_profile_to_cytof_cell_map, sample_over_cytof_count = self.cci.cytof_label_propagation(
                _a.T.copy(deep=True)), self.cci.get_cytof_count_per_sample()

        cell_abundance, res_as_cytof = get_results(_a, _b, X, full_profile_to_cytof_cell_map,**kwargs)
        cellMax_cell_abundance, cellMax_res_as_cytof = get_cell_mix_results(_a, _b, X, with_cellMix,
                                                                            full_profile_to_cytof_cell_map)

        display_result(cell_abundance, res_as_cytof, as_heatmap)
        display_result(cellMax_cell_abundance, cellMax_res_as_cytof, as_heatmap, " - CellMix")

        if with_cytof:
            sns.heatmap(sample_over_cytof_count.sort_index())
            plt.title("cytof results")
            plt.show()
        if X is not None:
            print(X.style.set_caption('known proportion'))
        return res_as_cytof,sample_over_cytof_count,cell_abundance

    def run_models_over_preProcess(self,list_of_preprocess,A,B,X=None, with_cellMix=True, as_heatmap=True, with_cytof=True):
        _all = "all" in list_of_preprocess

        if "naive" in list_of_preprocess  :
            self._calc_and_display_with_cellmix(A, B,X,with_cellMix,as_heatmap,with_cytof)

        if "naive discriminative" in list_of_preprocess :
            print("--------------naive discriminative-----------------")
            _A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
            _A, _B = PP_proteins.pp_naive_discriminative_proteins(_A, _B)
            self._calc_and_display_with_cellmix(_A,_B,X,with_cellMix,as_heatmap,with_cytof)

        if "binary_occurrence" in list_of_preprocess :
            print("--------------binary-----------------")
            _A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
            _A,_B = PP_proteins.pp_binary_occurrence(_A,_B)
            self._calc_and_display_with_cellmix(_A,_B,X,with_cellMix,as_heatmap,with_cytof)

        if "margin quantile" in list_of_preprocess :
            print("--------------margin quantile-----------------")
            _A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
            _A, _B = PP_proteins.pp_margin_quantile(_A, _B)
            self._calc_and_display_with_cellmix(_A, _B, X, with_cellMix, as_heatmap, with_cytof)

        if "under quantile" in list_of_preprocess :
            print( "--------------under quantile-----------------" )
            _A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
            _A, _B = PP_proteins.pp_under_quantile(_A, _B)
            self._calc_and_display_with_cellmix(_A, _B, X, with_cellMix, as_heatmap, with_cytof)

        if "entropy" in list_of_preprocess :
            print("-----------------entropy--------------------")
            __A, _B = PP_proteins.pp_clean_irrelevant_proteins(A, B)
            _A, _B = PP_proteins.pp_entropy_based(_A, _B)
            self._calc_and_display_with_cellmix(_A, _B, X, with_cellMix, as_heatmap, with_cytof)
