import pandas as pd
import csv

def data_integration(folder_location, file_names):
       log_spectrodetail = pd.read_csv((folder_location + '/' + file_names[0]))
       log_spectrok_pkpaper = pd.read_csv((folder_location + '/' + file_names[1]))
       mst_ink = pd.read_csv((folder_location + '/' + file_names[2]))
       mst_paper = pd.read_csv((folder_location + '/' + file_names[3]))
       udt_imx_bom = pd.read_csv((folder_location + '/' + file_names[4]))
       print("five data read done")

def data_analysis(folder_location, input_file_name, output_file_name):
       df = pd.read_csv(folder_location + '/' + input_file_name)

       nrows, ncols = df.shape
       
       print(len(df))
       print(df.shape)
       print(df.index)
       print(df.describe)
       
       analized_df = df.describe()
       analized_df.drop(analized_df.columns[0], axis=1)
       analized_df.insert(0, 'prop', ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
       analized_df.to_csv(folder_location + '/' + output_file_name, sep=',', index = False)
       # df.describe().to_csv(folder_location + '/' + output_file_name, sep=',', index = False)
