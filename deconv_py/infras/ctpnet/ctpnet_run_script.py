import pandas as pd
import ctpnet
import fileinput
import os
import sys
import io



de_line =''
for line in  fileinput.input():
    de_line += line

# de_line ='data_frame:C:\\Repos\\deconv_py\\deconv_py\\infras\\ctpnet/4fd.csv;weights_path:C:\\Repos\\deconv_py\\deconv_py\\infras\\ctpnet\\cTPnet_weight_24'

data_frame_path = de_line.split("data_frame:")[1].split(";weights_path:")[0]
weights_path = de_line.split("weights_path:")[1]

mrna_data_frame = pd.read_csv(data_frame_path)
mrna_data_frame = mrna_data_frame.drop(columns = mrna_data_frame.columns[0]).set_index(mrna_data_frame.columns[1])

imputed_proteins = ctpnet.predict.predict(mrna_data_frame,weights_path)

tmp_file_name = f"res_{os.path.basename(data_frame_path)}"
dir_name = os.path.dirname(data_frame_path)
result_path = os.path.join(dir_name,tmp_file_name)

imputed_proteins.to_csv(result_path)
print(f"**result**{result_path}**result**".encode("utf-8"))








