from deconv_py.infras.cytof_data.cytof_cell_count_infra import CytofCellCountInfra
from infras.deconv_data_frame import DecovAccessor
from deconv_py.models.cell_proportions.basic import BasicDeconv
from deconv_py.preprocess.cell_specifics.pp_agg_to_specific_cells import PpAggToSpecificCells
# from deconv_py.measures.cell_proportions_measures.cell_proportions_measure import CellProportionsMeasure
from infras.ctpnet.ctpnet_coordinator import CtpNetCoordinator

import itertools
from sklearn import pipeline
import pandas as pd
import uuid
import os
from tqdm import tqdm
import pickle
import hashlib
from functools import partial

class PipelineDeconv():
    def __init__(self,hyper_configuration,hyper_measure_configuration):
        self._hyper_configuration = hyper_configuration
        self._hyper_measure_configuration = hyper_measure_configuration

        self._static_configurations = list(self._build_static_configurations(self._hyper_configuration))
        self._pipeline_gen = self._build_pipelines(self._static_configurations)

        self._static_measure_configurations = list(self._build_static_configurations(self._hyper_measure_configuration))
        self._pipeline_measure_gen = self._build_pipelines(self._static_measure_configurations)

        self._ctpnet_coordinator = CtpNetCoordinator()

    #region public functions

    def run_pipeline(self,A,B,X):
        result_summary = []
        with tqdm(total=len(self._static_configurations)) as pbar:

            for pip in self._pipeline_gen:
                A = A.copy(deep=True)
                A.deconv.keep_original_data([A, B])

                try:
                    result = pip.predict([A, B])
                except Exception as e:
                    print(f"pipeline is : {pip}")
                    raise Exception(e)

                if result is None :
                    pbar.update(1)
                    continue

                pip_guid = uuid.uuid1().fields[0]
                pip_summary = self._get_pip_summary(pip)

                for measure in self._pipeline_measure_gen:
                    _pip_summary = self._measure(result, X, measure)
                    pip_summary = {**_pip_summary, **pip_summary}

                pip_summary["uuid"] = pip_guid
                result_summary.append(pip_summary)

                #we need to reset the generetor  - make it a list cause the weirdst bug ever
                self._pipeline_measure_gen = self._build_pipelines(self._static_measure_configurations)

                params_guid = f"{pip_guid}%" +"%".join([f"{k.split('_')[-1]}_{v}" for k,v in pip_summary.items()])
                _result_guid, _X_guid = f"result-{params_guid}.pkl",f"known-{params_guid}.pkl"

                with open(r"C:\Repos\deconv_py\deconv_py\experiments\archive\archive.txt", "a") as f:
                    f.write(_result_guid)
                    f.write("\n")
                    f.write(_X_guid)
                    f.write("\n")
                    f.write("\n")

                _X_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive",f"known-{pip_guid}.pkl")
                _result_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive",f"result-{pip_guid}.pkl")

                result.to_pickle(_result_path)
                X.to_pickle(_X_path)

                pbar.update(1)

        return pd.DataFrame(result_summary)

    def run_cytof_pipeline(self,A,B,per_cell_analysis = False,with_cache = True,cache_specific_signature = None,with_tree_cache = True):
        '''
        run pipeline where the measure is on the cytof data.there are some difference because of the label propagation
        for cytof pipeline : the hyper params configuration should have "Cytof_X_Building" step (with empty Transformer)
        ,so the code can build the cytof X in the right place.
        :param A: profile data
        :param B: cytof mixture data
        :return: measure results on cytof data
        '''

        result_summary = []

        with tqdm(total=len(self._static_configurations)) as pbar:

            for pip in self._pipeline_gen:
                pip_step_str = "_".join([str(s) for s in pip.steps])
                _func = partial(self._run_pipe_predict,A,B,pip,with_tree_cache,cache_specific_signature)
                result, X = self._run_or_load(pip_step_str,_func,with_cache,cache_specific_signature=cache_specific_signature)

                if result is None :
                    pbar.update(1)
                    continue

                pip_guid = uuid.uuid1().fields[0]
                pip_summary = self._get_pip_summary(pip)

                for measure in self._pipeline_measure_gen:
                    if per_cell_analysis :
                        measure_per_cell_str = f"percell_{str(measure.steps)}_{pip_step_str}"
                        _measure_per_cell_func = partial(self._measure_per_cell,result,X,measure)
                        _pip_summary = self._run_or_load(measure_per_cell_str,_measure_per_cell_func,with_cache,cache_specific_signature=cache_specific_signature)
                    else :
                        measure_str = f"{str(measure.steps)}_{pip_step_str}"
                        _measure_func = partial(self._measure, result, X, measure)
                        _pip_summary = self._run_or_load(measure_str, _measure_func,cache_specific_signature=cache_specific_signature)

                    pip_summary = {**_pip_summary, **pip_summary}

                pip_summary["uuid"] = pip_guid
                result_summary.append(pip_summary)

                #we need to reset the generetor  - make it a list cause the weirdest bug ever
                self._pipeline_measure_gen = self._build_pipelines(self._static_measure_configurations)

                params_guid = f"{pip_guid}%" +"%".join([f"{k.split('_')[-1]}_{v}" for k,v in pip_summary.items()])
                _result_guid, _X_guid = f"result-{params_guid}.pkl",f"known-{params_guid}.pkl"

                with open(r"C:\Repos\deconv_py\deconv_py\experiments\archive\archive.txt", "a") as f:
                    f.write(_result_guid)
                    f.write("\n")
                    f.write(_X_guid)
                    f.write("\n")
                    f.write("\n")

                _X_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive",f"known-{pip_guid}.pkl")
                _result_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive",f"result-{pip_guid}.pkl")

                if per_cell_analysis :
                    _signature_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive",f"signature-{pip_guid}.pkl")

                    with open(_signature_path,"wb") as f :
                        pickle.dump(pip.steps, f)

                result.to_pickle(_result_path)
                X.to_pickle(_X_path)

                pbar.update(1)

        return pd.DataFrame(result_summary)

    def transform_cytof_pipeline(self,A,B,with_label_prop = False):
        result_summary = []

        with tqdm(total=len(self._static_configurations)) as pbar:
            for pip in self._pipeline_gen:
                is_with_label_prop = pip.named_steps['Cytof_X_Building'].with_label_prop

                A = A.copy(deep= True)

                A.deconv.keep_original_data([A, B])
                result = pip.predict([A, B])
                before_prediction  = pip.transform([A,B])
                if result is None :
                    pbar.update(1)
                    continue

                if is_with_label_prop :
                    result , X = self._get_results_and_known_cytof(result, pip, True)
                else :
                    result , X = self._get_results_and_known_cytof(result, pip, False)

                #we need to reset the generetor  - make it a list cause the weirdst bug ever
                self._pipeline_measure_gen = self._build_pipelines(self._static_measure_configurations)

                result_summary.append([result,before_prediction,X])
                pbar.update(1)

        return result_summary

    def transform_specific_cytof_pipe(self,A, B, pip_steps):
        A = A.copy(deep=True)

        pip = pipeline.Pipeline(pip_steps)
        A.deconv.keep_original_data([A, B])

        result = pip.predict([A, B])
        before_prediction = pip.transform([A, B])

        result, X = self._get_results_and_known_cytof(result, pip, False)

        return [result, before_prediction, X]

    def load_results_from_archive(self,pkl_uuid,only_signature = False):
        dir_path = r"C:\Repos\deconv_py\deconv_py\experiments\archive"
        r_x_results = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".pkl") and (str(pkl_uuid) in filename):
                if only_signature :
                    if "signature" in filename :
                        with open(os.path.join(dir_path,filename),"rb") as f :
                            _sig = pickle.load(f)
                            r_x_results["signature"] = _sig
                    else :
                        continue

                if "result" in filename :
                    _r = pd.read_pickle(os.path.join(dir_path,filename))
                    r_x_results["result"] = _r
                if "known" in filename :
                    _x = pd.read_pickle(os.path.join(dir_path,filename))
                    r_x_results["known"] = _x

        return r_x_results

    def run_multi_signature_pipeline(self,A,B,meta_results,cpm,bd = None):
        if bd is None :
            bd = BasicDeconv()

        agg_spec_cells = PpAggToSpecificCells()

        cells_corr_columns = [c for c in meta_results.columns if "cellcorr" in c]
        relevent_meta = meta_results[cells_corr_columns]

        idx_best_sig_per_cell = relevent_meta.idxmax()

        best_uuids = {}
        for cell, idx in idx_best_sig_per_cell.items():
            best_uuids[cell] = meta_results.iloc[idx]["uuid"]

        all_sigs = []
        all_mixs = []
        for cell, _uuid in best_uuids.items():
            pipe_steps = self.load_results_from_archive(_uuid,only_signature = True)["signature"]
            _, before_prediction, X = self.transform_specific_cytof_pipe(A, B, pipe_steps)
            name = f"corr{cell.split('corr')[1].strip('Mean')}"
            _sig = before_prediction[0].copy(deep=True)
            _mix = before_prediction[1].copy(deep=True)

            if "NOT_" in _sig.columns[1]:
                _sig = \
                agg_spec_cells.transform([_sig.rename(columns={col: f"{col}_01" for col in _sig.columns}), None])[0]
                _sig = _sig.rename(columns={col: col.split("_01")[0] for col in _sig.columns})

            _sig["cell_ind"] = name
            _sig = _sig.set_index("cell_ind", append=True)

            _mix["cell_ind"] = name
            _mix = _mix.set_index("cell_ind", append=True)

            all_sigs.append(_sig)
            all_mixs.append(_mix)

        multi_signature = pd.concat(all_sigs)
        multi_mix = pd.concat(all_mixs)

        # now we found the results of bouth cell "types"
        result = bd.fit([multi_signature, multi_mix])

        measure_results = cpm.transform([result, X])
        pip_summary = measure_results

        pip_guid = uuid.uuid1().fields[0]
        pip_summary["uuid"] = pip_guid

        _X_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive", f"known-{pip_guid}.pkl")
        _result_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\archive", f"result-{pip_guid}.pkl")

        result.to_pickle(_result_path)
        X.to_pickle(_X_path)

        return pip_summary

    #endregion

    #region private function

    def _run_or_load(self,name_str,function,with_cache = True,cache_specific_signature=None):
        if cache_specific_signature is not None :
            name_str += cache_specific_signature

        cache_guid = hashlib.md5(name_str.encode()).hexdigest()
        _cache_path = os.path.join(r"C:\Repos\deconv_py\deconv_py\experiments\pipe_cache", f"{cache_guid}.pkl")

        res = None
        if os.path.exists(_cache_path) and with_cache:
            with open(_cache_path,"rb") as f :
                res = pickle.load(f)
        else:
            res = function()

            if with_cache :
                with open(_cache_path,'wb') as f :
                    pickle.dump(res,f)

        return res

    def _measure(self,result,X,measure):
        return measure.transform([result, X])

    def _measure_per_cell(self,result,X,measure) :
        _pip_summary = {}
        for cell in result.index:
            measure_results = measure.transform([result.loc[cell], X.loc[cell]])
            if measure_results is None:
                continue
            _pip_summary = {**measure_results, **_pip_summary}

        measure_results = measure.transform([result, X])
        _pip_summary = {**measure_results, **_pip_summary}
        return _pip_summary

    def _run_pipe_predict(self,A,B,pip,with_tree_cache,cache_specific_signature) :
        is_with_label_prop = pip.named_steps['Cytof_X_Building'].with_label_prop
        is_must_contain_sp = pip.steps[-1][1].get_params()["weight_sp"] or pip.steps[-1][1].get_params()["em_optimisation"]

        A = A.copy(deep=True)
        A.deconv.keep_original_data([A, B])

        if with_tree_cache :
            A.deconv.sign_transformer_to_df(cache_specific_signature)

        if is_must_contain_sp :
            A.deconv.set_must_contain_gene(must_contain_genes=self._get_sp_genes_idx(A))

        result = pip.predict([A, B])
        # try:
        #     result = pip.predict([A, B])
        # except Exception as e:
        #     print(f"pipeline is : {pip}")
        #     result = pip.predict([A, B])
        #     raise Exception(e)

        if result is None :
            return None,None

        #TODO : change to new cytof
        if is_with_label_prop:
            result, X = self._get_results_and_known_cytof(result, pip, True)
        else:
            result, X = self._get_results_and_known_cytof(result, pip, False)

        return result, X

    def _get_pip_summary(self,pip):
        not_relevent_params = ['PpEmpty_keep_labels','PpEmpty__labels','BasicDeconv_cmc','BasicDeconv_result']

        params_dicts = [{f"{str(function).split('(')[0]}_{k}": str(v)[:7] for k, v in function.__dict__.items()} for
                        step, function in pip.steps]
        params_dic = dict((key,d[key]) for d in params_dicts for key in d)

        for p in not_relevent_params :
            if p in params_dic.keys() :
                params_dic.pop(p)

        return params_dic

    def _get_results_and_known_cytof(self,result, pip, with_label_prop):
        cluster_info_path = r"C:\Repos\deconv_py\deconv_py\infras\cytof_data\raw_data/CyTOF.features.and.clusters.info.xlsx"
        cytof_data_path = r"C:\Repos\deconv_py\deconv_py\infras\cytof_data\raw_data/filtered.esetALL.CyTOF.abundance.only.xlsx"
        cci = CytofCellCountInfra(cluster_info_path = cluster_info_path,cytof_data_path = cytof_data_path)

        if not with_label_prop :
            return cci.return_mass_and_cytof_not_none_cells_counts(result,filter_by_version="")

        _A = pip.named_steps["Cytof_X_Building"]._labels["A"]

        self.propagation = cci.cytof_label_propagation(_A)
        return cci.return_mass_and_cytof_not_none_cells_counts(result, self.propagation,filter_by_version = "")

    def _build_static_configurations(self, hyper_configuration):
        static_configuration = {}

        step_configuration = []
        for step in hyper_configuration:
            functions_combs = []
            for s in step["steps"]:
                func_name = s["function_name"]
                func = s["function"]
                params = s["params"]
                all_params_comb = list(itertools.product(*[[(k, vv) for vv in v] for k, v in params.items()]))
                for params_comb in all_params_comb:
                    functions_comb = [func_name, func, params_comb]
                    functions_combs.append(functions_comb)

            step_configuration.append(functions_combs)
        #         static_configuration[step["step_name"]] =  functions_combs
        return [list(zip([s["step_name"] for s in hyper_configuration], config)) for config in
                itertools.product(*step_configuration)]

    def _build_pipelines(self, static_configurations):
        for static_conf in static_configurations:
            pipeline_steps = []
            params_sklearn_set = {}
            for step in static_conf:
                function_name = step[1][0]
                function_inst = step[1][1]
                function_param = step[1][2]

                pipeline_steps.append((function_name, function_inst))
                params_sklearn_set.update({f"{function_name}__{p[0]}": p[1] for p in function_param})

            curr_ppline = pipeline.Pipeline(pipeline_steps)
            curr_ppline.set_params(**params_sklearn_set)
            yield curr_ppline

    def _fix_to_corospanding_cell_names(self,result,X):
        if X.index[0] in result.index :
            return X

        cells_res = result.index.to_list()
        cells_x = X.index.to_list()

        mapping = {}

        for cr in cells_res:
            for cx in cells_x:
                if (cx in cr) or (cr in cx):
                    mapping[cx] = cr

        return X.rename(index=mapping)

    def _get_sp_genes_idx(self,cell_profile):
        sp_genes_list = self._ctpnet_coordinator.get_sp_genes_list()

        return cell_profile.loc[cell_profile.index.get_level_values(1).isin(sp_genes_list)].index

    #endregion