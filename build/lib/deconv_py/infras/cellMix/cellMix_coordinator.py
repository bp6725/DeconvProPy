import feather
import os
import sys
import time
import pandas as pd
import uuid
from shutil import copyfile,copy2,copy,rmtree
import feather
import subprocess
from datetime import datetime as dt

class CellMixCoordinator() :

    def cell_prop_with_bash(self,B : pd.DataFrame, A:pd.DataFrame,wd_path : str = None,bash_file_name : str = 'ged_script.R'):
        tmp_folder_path = self._create_unique_folder()
        bash_path = self._cp_bash_to_folder(tmp_folder_path,bash_file_name)
        self._write_df_as_feather(B,os.path.join(tmp_folder_path,'mixture'))
        self._write_df_as_feather(A,os.path.join(tmp_folder_path,'sign'))

        self._exacute_bash(bash_path)

        while (not self._is_bash_finish(tmp_folder_path)) :
            time.sleep(1)

        res = self._load_results(tmp_folder_path)
        rmtree(tmp_folder_path)

        return res

    def _create_unique_folder(self):
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        curr_dir = r"C:\Repos\deconv_py\deconv_py\infras\cellMix"

        _now = dt.now()
        unique_folder_name = f"{_now.second}_{_now.hour}_{_now.day}_{_now.month}_{_now.year}_{uuid.uuid1().hex[0:3]}"
        dir_path = os.path.join(curr_dir,unique_folder_name)
        os.mkdir(dir_path)
        return dir_path

    def _cp_bash_to_folder(self,folder_to,file):
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        curr_dir = r"C:\Repos\deconv_py\deconv_py\infras\cellMix"
        file_path = os.path.join(curr_dir, file)
        copy(file_path,folder_to)
        return os.path.join(folder_to,file)

    def _write_df_as_feather(self,file,to_folder_path) :
        if isinstance(file,pd.DataFrame) :
            feather.write_dataframe(file, to_folder_path)

    def _exacute_bash(self,bash_path) :
        rscript_exe_path = os.path.join(os.environ['R_HOME'],r"bin\\Rscript.exe")
        subprocess.call([rscript_exe_path, '--vanilla', bash_path,os.path.dirname(bash_path)], shell=True)

    def _is_bash_finish(self,folder_path):
        _sucsess_finish_file_indc_path = os.path.join(folder_path,'run_finished.txt')
        _failed_finish_file_indc_path = os.path.join(folder_path, 'run_failed.txt')

        if os.path.exists(_failed_finish_file_indc_path) :
            with open(_failed_finish_file_indc_path,'r') as f:
                err = f.read()
            raise Exception(err)
        return os.path.exists(_sucsess_finish_file_indc_path)

    def _load_results(self,folder_path) :
        result_path = os.path.join(folder_path, 'ged_res')
        res  = pd.read_csv(result_path)
        return res


if __name__ == '__main__':
    A = pd.read_pickle('A_tmp')
    B = pd.read_pickle('B_tmp')
    #
    cmc = CellMixCoordinator()
    res = cmc.cell_prop_with_bash(B,A)

    # cmc._exacute_bash(r"C:\Repos\deconv_py\deconv_py\infras\cellMix\20_16_22_10_2019_760\ged_script.R")
    # print(res)

