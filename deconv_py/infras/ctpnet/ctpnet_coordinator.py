import pandas as pd
import subprocess
from subprocess import  Popen, PIPE, STDOUT
import sys
import os
import shlex
import uuid
import io
import numpy as np
from deconv_py.infras.global_utils import GlobalUtils
import pickle as pkl

MASS_CELL_NAMES_TO_HAMRNA_CELL_MAPPING =  {'BCellmemory': 'memory B-cell',
                                    'BCellnaive' : 'naive B-cell',
                                    'BCellplasma' : None,
                                    'CD4TCellmTregs' : 'T-reg',
                                    'CD4TCellnaive':'naive CD4 T-cell',
                                    'CD4TCellnTregs':'T-reg',
                                    'CD4TCellTcm' : 'memory CD4 T-cell' ,
                                    'CD4TCellTem' : None,
                                    'CD4TCellTemra' :None,
                                    'CD4TCellTh1':None,
                                    'CD4TCellTh17':None,
                                    'CD4TCellTh2':None,
                                    'CD8TCellnaive':'naive CD8 T-cell',
                                    'CD8TCellTcm' : 'memory CD8 T-cell',
                                    'CD8TCellTem' : None ,
                                    'CD8TCellTemra' : None,
                                    'DendriticCD1c' : None,
                                    'DendriticCD304' : None,
                                    'Erythrocytes' : None,
                                    'Monocytesclassical': 'classical monocyte' ,
                                    'Monocytesintermediate' : 'intermediate monocyte',
                                    'Monocytesnonclassical': 'non-classical monocyte',
                                    'NKCellsCD56bright' : 'NK-cell',
                                    'NKCellsCD56dim': None,
                                    'Thrombocytes' : None,
                                    'Granulocyteseosinophils' : "eosinophil",
                                    'GranulocytesBasophil' : "basophil"}


class CtpNetCoordinator():
    def __init__(self,mrna_data_path =r"C:\Repos\deconv_py\deconv_py\infras\ctpnet\human_atlas_for_ctpnet_prediction.csv",
                 genes_list = r"C:\Repos\deconv_py\deconv_py\infras\ctpnet\genes_list.txt",
                 weights_path= r"C:\Repos\deconv_py\deconv_py\infras\ctpnet\cTPnet_weight_24",
                 run_script_path = r"C:\\Repos\\deconv_py\\deconv_py\\infras\\ctpnet\\ctpnet_run_script.py",
                 env_path = r"C:\Users\Shenorr\.conda\envs\CTP_env\python.exe",
                 path_of_cached_imputed_proteins = None):

        self.env_path = env_path
        self.weights_path = weights_path
        self.run_script_path = run_script_path
        self.mrna_data_path = mrna_data_path
        self.genes_list = genes_list

        if path_of_cached_imputed_proteins is None :
            self.imputed_proteins = None
        else :
            with open(path_of_cached_imputed_proteins,"rb") as f :
                self.imputed_proteins = pkl.load(f)

    def return_imputed_proteins_for_cells(self,cells_list,cells_name_mapping = MASS_CELL_NAMES_TO_HAMRNA_CELL_MAPPING):
        if self.imputed_proteins is None :
            ctpnet_result_df = self.calculate_ctpnet_results()
            self.imputed_proteins = ctpnet_result_df
        else :
            ctpnet_result_df = self.imputed_proteins

        ctpnet_to_run_time_cells_mapping = self._return_cells_mapping(cells_list,cells_name_mapping)

        relevant_ctpnet_results = ctpnet_result_df[[cell for cell in ctpnet_to_run_time_cells_mapping.keys() if cell is not None]]
        relevant_ctpnet_results = relevant_ctpnet_results.rename(columns = ctpnet_to_run_time_cells_mapping)

        return np.exp(relevant_ctpnet_results)

    def get_sp_genes_list(self):
        if self.imputed_proteins is None :
            ctpnet_result_df = self.calculate_ctpnet_results()
            self.imputed_proteins = ctpnet_result_df
        else :
            ctpnet_result_df = self.imputed_proteins

        return ctpnet_result_df.index.to_list()

    def calculate_ctpnet_results(self):
        if self.imputed_proteins is not None :
            print("imputed protein is already calculated")

        mrna_data = self._build_relevent_mrna_data()
        mrna_file_path = self._save_temp_mrna_data(mrna_data)

        input_encoding = f"data_frame:{mrna_file_path};weights_path:{self.weights_path}".encode('utf-8')
        ctpnet_result_df,tmp_file_path = self._get_ctpnet_results(input_encoding)
        ctpnet_result_df = ctpnet_result_df.set_index(ctpnet_result_df.columns[0])

        self.imputed_proteins = ctpnet_result_df
        os.remove(mrna_file_path)
        os.remove(tmp_file_path)

        return ctpnet_result_df

    def _return_cells_mapping(self,cells_list,cells_name_mapping):
        run_time_cells_to_Mass_cells_mapping = GlobalUtils.get_corospanding_cell_map_from_lists(cells_list,cells_name_mapping.keys())
        ctpnet_to_run_time_cells_mapping = {cells_name_mapping[run_time_cells_to_Mass_cells_mapping[run_time_cell]] : run_time_cell for run_time_cell in cells_list }

        return ctpnet_to_run_time_cells_mapping

    def _build_relevent_mrna_data(self):
        required_ctpnet_gene_list = self._load_gene_list()
        raw_mrna_df = pd.read_csv(self.mrna_data_path)
        raw_mrna_df = raw_mrna_df.set_index("Unnamed: 0")

        # if len(pd.Index(required_ctpnet_gene_list).difference(raw_mrna_df.index)) >0 :
        #     raise Exception("mrna matrix dont have all the genes")

        return raw_mrna_df

    def _save_temp_mrna_data(self,mrna_data):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        guid = uuid.uuid1().hex[0:3]
        path_to_save = f"{dir_path}/{guid}.csv"
        mrna_data.to_csv(path_to_save)
        return path_to_save

    def _get_ctpnet_results(self,input_encoding):
        # command = f"source activate environment-{self.env_path} && python {self.run_script_path}"
        command = f"python {self.run_script_path}"
        args = shlex.split(command)
        p = Popen(args, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        stdout = p.communicate(input=input_encoding)
        p.wait()

        ctpnet_result_path = stdout[0].decode("utf-8").split("**result**")[1]
        ctpnet_result_df = pd.read_csv(ctpnet_result_path)

        return ctpnet_result_df,ctpnet_result_path

    def _load_gene_list(self):
        gene_list = []
        with open(self.genes_list,"r") as f :
            for line in f.readlines() :
                gene_list.append(line.split(",")[1].split('"')[1])
        return gene_list

if __name__ == '__main__':
    ctpc = CtpNetCoordinator()
    res = ctpc.calculate_ctpnet_results()
    print(res)