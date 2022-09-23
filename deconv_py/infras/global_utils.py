import pandas as pd
import re
from itertools import product
import mygene

_low_case_mixture_format = "\d\d_v\d"
_upper_case_mixture_format = "HR-\d\d-V\d"

class GlobalUtils() :
    def __init__(self):
        pass

    @staticmethod
    def get_corospanding_mixtures_map(source :pd.DataFrame, target:pd.DataFrame):
        source_columns = source.columns
        target_columns = target.columns

        re_low_case_mixture = re.compile(_low_case_mixture_format)
        re_upper_case_mixture = re.compile(_upper_case_mixture_format)

        mapping = {}
        if re_low_case_mixture.match(target_columns[0]) :
            for s_mixture in source_columns :
                mapping[s_mixture] = _low_case_mixture_format.replace("\d", '{}').format(*re.findall("\d", s_mixture)[0:3])

        if re_upper_case_mixture.match(target_columns[0]) :
            for s_mixture in source_columns :
                mapping[s_mixture] = _upper_case_mixture_format.replace("\d", '{}').format(*re.findall("\d", s_mixture)[0:3])

        return mapping

    @staticmethod
    def get_corospanding_mixtures_map_from_lists(source, target):
        source_columns = source
        target_columns = target

        re_low_case_mixture = re.compile(_low_case_mixture_format)
        re_upper_case_mixture = re.compile(_upper_case_mixture_format)

        mapping = {}
        if re_low_case_mixture.match(target_columns[0]):
            for s_mixture in source_columns:
                mapping[s_mixture] = _low_case_mixture_format.replace("\d", '{}').format(
                    *re.findall("\d", s_mixture)[0:3])

        if re_upper_case_mixture.match(target_columns[0]):
            for s_mixture in source_columns:
                mapping[s_mixture] = _upper_case_mixture_format.replace("\d", '{}').format(
                    *re.findall("\d", s_mixture)[0:3])

        return mapping

    @staticmethod
    def get_corospanding_cell_map(source : pd.DataFrame,target:pd.DataFrame):
        source_columns = source.columns
        target_columns = target.columns

        return GlobalUtils.get_corospanding_cell_map_from_lists(source_columns,target_columns)


    @staticmethod
    def get_corospanding_cell_map_from_lists(source, target):
        source_target_comb = product(source, target)
        filt_source_target_comb = list(filter(lambda s_t: s_t[1] in s_t[0], source_target_comb))

        if not filt_source_target_comb :
            source_target_comb = product(source, target)
            filt_source_target_comb = list(filter(lambda s_t: s_t[0] in s_t[1], source_target_comb))


        return {s: t for (s, t) in filt_source_target_comb}

    @staticmethod
    def return_ens_id_to_gene_name(list_of_ids):
        mg = mygene.MyGeneInfo()
        ginfo = mg.querymany(list_of_ids, scopes='ensembl.gene')

        ensg_id_to_gene_name_mapping = {}
        for g in ginfo:
            if ("query" in g.keys()) and ("symbol" in g.keys()):
                ensg_id_to_gene_name_mapping[g["query"]] = g["symbol"]
        return ensg_id_to_gene_name_mapping