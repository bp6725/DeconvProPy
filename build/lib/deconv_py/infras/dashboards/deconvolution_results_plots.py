from deconv_py.infras.global_utils import GlobalUtils
from deconv_py.experiments.pipeline.pipeline_deconv import PipelineDeconv
from IPython.display import display, HTML
from infras.dashboards import exploration_cytof_plots as cytof_plots
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

class DeconvolutionResultsPlots():

    @staticmethod
    def describe_results(uuids,meta_results = None,with_per_mixture_plot = False,with_mixtures_pca = False) :
        _pipe = PipelineDeconv({},{})

        if type(uuids) is not list :
            uuids = [uuids]

        for uuid in uuids :
            if meta_results is not None :
                params = meta_results[meta_results["uuid"] == int(uuid)].T.copy(deep=True).dropna()
                print("params : ")
                display(HTML(params.to_html()))

            best_results_and_known = _pipe.load_results_from_archive(uuid)
            best_results=best_results_and_known["result"]
            best_known=best_results_and_known["known"]
            mapping = GlobalUtils.get_corospanding_mixtures_map(best_known,best_results)
            best_known = best_known.rename(columns=mapping)
            best_known = best_known[[col for col in mapping.values()]]

            mutual_col = best_known.columns.intersection(best_results.columns)
            best_results = best_results[mutual_col]
            best_known = best_known[mutual_col]

            print("mixtures : ")
            display(HTML(best_results.to_html()))

            print("mixtures details :")
            display(HTML(best_results.corrwith(best_known,method="spearman").describe().to_frame().to_html()))

            cytof_plots.plot_mass_to_cytof_scatter_all_on_one(best_results,best_known,best_results)
            if with_per_mixture_plot:
                cytof_plots.plot_mass_to_cytof_scatter(best_results, best_known, best_results)
            if with_mixtures_pca:
                DeconvolutionResultsPlots.plot_results_vs_known_pca(best_results, best_known)

    @staticmethod
    def plot_results_vs_known_pca(best_results, best_known):
        pca = PCA(n_components=2)
        pca.fit(pd.concat([best_results, best_known], axis=1).T)

        deconv_principalcomp = pca.transform(best_results.T)
        known_principalcomp = pca.transform(best_known.T)

        deconv_principalDf = pd.DataFrame(data=deconv_principalcomp
                                          , columns=['principal component 1', 'principal component 2'],
                                          index=best_results.columns)
        known_principalDf = pd.DataFrame(data=known_principalcomp
                                         , columns=['principal component 1', 'principal component 2'],
                                         index=best_known.columns)

        deconv_principalDf["color"] = "b"
        known_principalDf["color"] = "r"

        principalDf = deconv_principalDf.append(known_principalDf)
        fig = plt.figure(figsize=(25, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('blue - deconvolution result,red -  known proportions', fontsize=20)

        ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=principalDf['color'],
                   s=50)

        for mixture in range(deconv_principalcomp.shape[0]):
            deconv_point = deconv_principalcomp[mixture, :]
            known_point = known_principalcomp[mixture, :]
            plt.plot([deconv_point[0], known_point[0]], [deconv_point[1], known_point[1]], ':')

        for i, txt in enumerate(principalDf.index):
            ax.annotate(txt,
                        (principalDf['principal component 1'].iloc[i], principalDf['principal component 2'].iloc[i]))

    @staticmethod
    def plot_measures_results_graph(meta_results):
        _meta_results = meta_results.copy(deep=True)
        _meta_results = _meta_results.set_index("corrMean").sort_index(ascending=True)
        p_value = 0.15
        measures_cols = [col.split("Mean")[0] for col in _meta_results.columns if "Mean" in col]

        for measure, style in zip(measures_cols, ['-ro', '-bo', '-go']):
            measures_series = _meta_results[[measure + "Mean", measure + "Pval"]]
            measures_series[measures_series[measure + "Pval"] < p_value][measure + "Mean"].plot(style=style,
                                                                                                legend=True)

    @staticmethod
    def return_best_correlation_per_cell_type(meta_results):
        cells_corr_columns = [c for c in meta_results.columns if "cellcorr" in c]
        relevent_meta = meta_results[cells_corr_columns]

        return meta_results.iloc[relevent_meta.idxmax().values][["corrMean", "uuid"]]

    @staticmethod
    def return_best_result_sum_across_per_cell(meta_results):
        cells_corr_columns = [c for c in meta_results.columns if "cellcorr" in c]
        idx = meta_results[cells_corr_columns].fillna(-1).apply(lambda x: x.argsort().argsort()).sum(axis=1).argmax()
        return meta_results.iloc[idx]