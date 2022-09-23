import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from deconv_py.infras.global_utils import GlobalUtils
import random
from scipy import stats
from itertools import combinations,product
from scipy.stats import entropy
from scipy.spatial.distance import pdist,squareform
from skbio.stats.distance import mantel

class CellProportionsMeasure(TransformerMixin,BaseEstimator) :

    def __init__(self,how = "correlation",return_dict_measure_statas = True,with_pvalue = False,with_iso_test = False,correlation_method = "pearson"):
        self.how = how
        self.return_dict_measure_statas = return_dict_measure_statas
        self.with_pvalue = with_pvalue
        self.with_iso_test = with_iso_test
        self.correlation_method = correlation_method

    def transform(self, data, *_):
        deconv_result, known_results = data[0],data[1]

        if (type(deconv_result) is pd.Series)  :
            if (self.how == "correlation") :
                columns_mapping = GlobalUtils.get_corospanding_mixtures_map_from_lists(known_results.index.to_list(), deconv_result.index.to_list())
                known_results = known_results.rename(index=columns_mapping)
                measure = self.per_cell_correlation_measure(deconv_result,known_results)
                if self.return_dict_measure_statas :
                    _name = "cellcorr" + deconv_result.name.replace(" ","")
                    return self._statistics_over_per_cell_measure(measure,type=_name)
                return measure
            else :
                return None

        columns_mapping = GlobalUtils.get_corospanding_mixtures_map(known_results, deconv_result)
        known_results = known_results.rename(columns = columns_mapping)

        mu_mixtures = known_results.columns.intersection(deconv_result.columns)
        known_results = known_results[mu_mixtures].copy()
        deconv_result = deconv_result[mu_mixtures].copy()

        if self.how == "correlation" :
            if not any([(col in known_results.index.tolist()) for col in deconv_result.index]) :
                columns_mapping = GlobalUtils.get_corospanding_cell_map_from_lists(known_results.index.to_list(),
                                                                                       deconv_result.index.to_list())
                known_results = known_results.rename(index=columns_mapping)

            measure = self.correlation_measure(deconv_result,known_results,self.with_pvalue,self.with_iso_test)
            if self.return_dict_measure_statas :
                return self._statistics_over_measure(measure)
            return measure

        if self.how == "RMSE" :
            measure = self.RMSE_measure(deconv_result, known_results, self.with_pvalue,self.with_iso_test)
            if self.return_dict_measure_statas:
                return self._statistics_over_measure(measure,type="rmse",right_pval = False)
            return measure

        if self.how == "MI" :
            measure = self.mi_measure(deconv_result, known_results, self.with_pvalue,self.with_iso_test)
            if self.return_dict_measure_statas:
                return self._statistics_over_measure(measure, type="MI", right_pval=True)
            return measure

        if "entropy" :
            if deconv_result.sum().sum() == 0 :
                return {"entropy" : None}
            return {"entropy" : deconv_result.apply(lambda x: entropy(x)).mean()}

        if self.how == "groups" :
            measures = self.group_measure(deconv_result, known_results)

            if self.return_dict_measure_statas:
                res = {}
                for group_name,measure in measures.items():
                    stats = self._statistics_over_measure(measure)
                    res[group_name] = stats
                return res

            return measures

    def fit(self, *_):
        return self

    def correlation_measure(self, deconv_result, known_results,with_pvalue = False,with_iso_test = False):
        measure_function = lambda x,y : x.corrwith(y, axis=0, method=self.correlation_method)
        final_results = {}

        if with_pvalue :
            permutations = []
            known_results = known_results.copy(deep = True)
            col_list = known_results.columns.tolist()
            for i in range(99):
                permut_columns_map = {k:v for k,v in zip(col_list,random.sample(col_list, len(col_list)))}
                _known_results = known_results.rename(columns = permut_columns_map)
                _measure = measure_function(deconv_result,_known_results)
                permutations.append((_measure))
            measure = measure_function(deconv_result,known_results)

            final_results["measure"] = measure
            final_results["permuts"] = permutations

        if with_iso_test :
            series_wise_measure_function = lambda x,y :  x.corr(y,method = self.correlation_method)
            final_results["iso"] =  self.calc_isomorphism_score(deconv_result,known_results,series_wise_measure_function,right_pval = True)

        if with_iso_test or with_pvalue :
            return final_results

        return measure_function(deconv_result,known_results)

    def per_cell_correlation_measure(self, deconv_result, known_results,with_pvalue = False,with_iso_test = False):
        measure_function = lambda x, y: x.corr(y, method=self.correlation_method)
        final_results = {}

        if with_pvalue :
            permutations = []
            known_results = known_results.copy(deep = True)
            for i in range(99):
                _known_results_values = known_results.values
                _known_results = pd.Series(index=known_results.index ,data = np.random.permutation(_known_results_values))
                _measure = measure_function(deconv_result,_known_results)
                if _measure is np.nan :
                    _measure = 0
                permutations.append((_measure))
            measure = measure_function(deconv_result,known_results)

            if np.isnan(measure):
                measure = 0

            final_results["measure"] = measure
            final_results["permuts"] = permutations

            return final_results

        return measure_function(deconv_result,known_results)

    def RMSE_measure(self, deconv_result, known_results, with_pvalue = False,with_iso_test=False):
        def rmse_loss(deconv,known):
            return ((deconv - known) ** 2).mean() ** 0.5

        final_results = {}

        if with_pvalue :
            permutations = []
            known_results = known_results.copy(deep = True)
            col_list = known_results.columns.tolist()
            for i in range(99):
                permut_columns_map = {k:v for k,v in zip(col_list,random.sample(col_list, len(col_list)))}
                _known_results = known_results.rename(columns = permut_columns_map)
                _measure = rmse_loss(deconv_result,_known_results)
                permutations.append((_measure))
            measure = rmse_loss(deconv_result,known_results)

            final_results["measure"] =measure
            final_results["permuts"] = permutations

        if with_iso_test:
            series_wise_measure_function =lambda x,y :  ((x- y) ** 2).mean() ** 0.5
            final_results["iso"] = self.calc_isomorphism_score(deconv_result, known_results, series_wise_measure_function,right_pval = False)

        if with_iso_test or with_pvalue:
            return final_results

        return rmse_loss(deconv_result,known_results)

    def mi_measure(self, deconv_result, known_results, with_pvalue = False,with_iso_test = False):
        def mi_loss(deconv,known):
            measure_function = lambda x, y:( x.cov(y) / (x.var() + y.var()))

            res = pd.Series(index=deconv.columns)
            for mixture in deconv.columns:
                x = deconv[mixture]
                y = known[mixture]
                res.loc[mixture] = measure_function(x, y)
            return res

        final_results = {}
        if with_pvalue :
            permutations = []
            known_results = known_results.copy(deep = True)
            col_list = known_results.columns.tolist()
            for i in range(99):
                permut_columns_map = {k:v for k,v in zip(col_list,random.sample(col_list, len(col_list)))}
                _known_results = known_results.rename(columns = permut_columns_map)
                _measure = mi_loss(deconv_result,_known_results)
                permutations.append((_measure))
            measure = mi_loss(deconv_result,known_results)

            final_results["measure"] = measure
            final_results["permuts"] = permutations

        if with_iso_test:
            series_wise_measure_function =  lambda x, y: x.cov(y) / (x.var() + y.var())
            final_results["iso"] = self.calc_isomorphism_score(deconv_result, known_results, series_wise_measure_function,right_pval = True)

        if with_iso_test or with_pvalue:
            return final_results

        return mi_loss(deconv_result,known_results)

    def group_measure(self, deconv_result, known_results):
        def chunkify(lst, n):
            return [lst[(i * len(lst)) // n:((i + 1) * len(lst)) // n] for i in range(n)]

        X_per_chunk = chunkify(known_results.mean(axis=1).sort_values(), 3)

        res = {}
        for chunk, chunk_name in zip(X_per_chunk, ["low abundance", "mid abundance", "high abundance"]):
            result_chunk = deconv_result.loc[chunk.index]

            corr = self.correlation_measure(result_chunk, chunk)
            res[chunk_name] = corr
        return res

    def _statistics_over_measure(self,measure,only_mean = False,type = "corr",right_pval = True):
        if isinstance(measure,dict) :
            obs = measure["measure"]
            permuts = measure["permuts"]

            permuts_stats = []
            for permut in permuts:
                _m = self._statistics_over_measure(permut,only_mean =True)
                permuts_stats.append(_m)
            _m_obs = self._statistics_over_measure(obs,only_mean =True)
            pval = abs(int(right_pval) - stats.percentileofscore(permuts_stats, _m_obs) / 100)

            final_stats_with_pvalue = self._statistics_over_measure(obs,type = type)
            final_stats_with_pvalue[f"{type}Pval"] = np.round(pval,3)

            if self.with_iso_test :
                final_stats_with_pvalue[f"{type}Iso"] = measure["iso"]

            return final_stats_with_pvalue

        if only_mean:
            return np.round(measure.mean(), 3)

        mean = np.round(measure.mean(), 3)
        # std = np.round(measure.std(), 3)
        # non = np.round(measure.isna().sum() / measure.count() , 3)
        # return {f"{type}Mean": mean, f"{type}Std": std}

        return {f"{type}Mean": mean}

    def _statistics_over_per_cell_measure(self,measure,only_mean = False,type = "corr",right_pval = True):
        if isinstance(measure,dict) :
            obs = measure["measure"]
            permuts = measure["permuts"]

            permuts_stats = []
            for permut in permuts:
                _m = self._statistics_over_per_cell_measure(permut,only_mean =True)
                permuts_stats.append(_m)
            _m_obs = self._statistics_over_per_cell_measure(obs,only_mean =True)
            pval = abs(int(right_pval) - stats.percentileofscore(permuts_stats, _m_obs) / 100)

            final_stats_with_pvalue = self._statistics_over_per_cell_measure(obs,type = type)
            final_stats_with_pvalue[f"{type}Pval"] = np.round(pval,3)

            return final_stats_with_pvalue

        if only_mean:
            return np.round(measure, 3)

        mean = np.round(measure, 3)
        return {f"{type}Mean": mean}

    def calc_isomorphism_score(self,deconv, known, measure_function,right_pval):
        _known_distances_values =[]
        _deconv_distances_values =[]

        #building pairwise distance

        for mixture in deconv.columns :
            _known_data = known[mixture]
            _deconv_data = deconv[mixture]

            _s_known_distances_values =[]
            _s_deconv_distances_values =[]
            for s_mixture in deconv.columns :
                _s_known_data = known[s_mixture]
                _s_deconv_data = deconv[s_mixture]

                _s_known_distances_values.append(measure_function(_known_data,_s_known_data))
                _s_deconv_distances_values.append(measure_function(_deconv_data,_s_deconv_data))

            _known_distances_values.append(_s_known_distances_values)
            _deconv_distances_values.append(_s_deconv_distances_values)

        known_distances = pd.DataFrame(index=known.columns,columns=known.columns,data=_known_distances_values)
        deconv_distances = pd.DataFrame(index=deconv.columns, columns=deconv.columns, data=_deconv_distances_values)

        if right_pval :
            worst_dist = min(deconv_distances.min().min(),known_distances.min().min())
        else :
            worst_dist = min(deconv_distances.max().max(), known_distances.max().max())

        known_distances = known_distances.fillna(worst_dist)
        deconv_distances = deconv_distances.fillna(worst_dist)

        known_distances = abs(known_distances - known_distances.iloc[0,0]).round(2)
        deconv_distances = abs(deconv_distances - deconv_distances.iloc[0, 0]).round(2)

        np.fill_diagonal(known_distances.values, 0)
        np.fill_diagonal(deconv_distances.values, 0)

        iso_res = mantel(known_distances,deconv_distances)[0]

        return np.round(iso_res,3)



