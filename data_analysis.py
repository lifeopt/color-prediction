import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from copy import deepcopy

print_tag = False
plot_tag = True

def data_integration(folder_location, file_names):
       log_spectrodetail = pd.read_csv((folder_location + '/' + file_names[0]))
       log_spectrok_pkpaper = pd.read_csv((folder_location + '/' + file_names[1]))
       mst_ink = pd.read_csv((folder_location + '/' + file_names[2]))
       mst_paper = pd.read_csv((folder_location + '/' + file_names[3]))
       udt_imx_bom = pd.read_csv((folder_location + '/' + file_names[4]))
       print("five data read done")

def delete_outlier(df, folder_location, output_file_name_deleted, output_file_name_new):  # 아웃라이어제거
       pick_paper_SPR = pd.DataFrame(df.loc[:, 'PAPER_SPECR_01':'PAPER_SPECR_31'])
       pick_all_SPR = pd.DataFrame(df.loc[:, 'SPECR_01':'SPECR_31'])
       check_paper_SPR = (pick_paper_SPR > 100).any(1)      # 행에 100넘는거 있는지 검사
       check_all_SPR = (pick_all_SPR > 100).any(1)      # 행에 100넘는거 있는지 검사
       preprocessed_df = pd.DataFrame(np.logical_and(check_paper_SPR, check_paper_SPR))
       mask = np.logical_and(check_paper_SPR, check_paper_SPR)
       mask_index = np.where(mask)
       deleted_df = df.loc[mask_index[0]]    # 삭제된 정보
       new_df = df.drop(index = mask_index[0], axis = 0)       # 행 삭제
       df_inks = df.iloc[:, 70:125]  # inks data
       # df_inks.dropna()       
       deleted_df.to_csv(folder_location + '/' + output_file_name_deleted, sep=',', index = False)
       new_df.to_csv(folder_location + '/' + output_file_name_new, sep=',', index = False)
       return new_df

def fundamental_analysis(df, folder_location, output_file_name):
       analized_df = df.describe()
       nrows, ncols = df.shape
       skewness = ['-']
       for i in range(ncols):
              if i==0: continue    # 첫컬럼제외
              skewness.append(df.iloc[:,i].skew())
       nrows, ncols = analized_df.shape
       analized_df.loc[nrows] = skewness
       analized_df.insert(0, 'property', ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max','sckewness'])
       analized_df.to_csv(folder_location + '/' + output_file_name, sep=',', index = False)       # 기초 통계 결과출력
       
def show_box_plot_paper(df):       # box plot for paper
       plt.figure(plt.gcf().number)
       temp_plot = sns.boxplot(data = df.loc[:, ['WIDTH', 'HEIGHT', 'WEIGHT']])

def show_box_plot_spectral(df_spec):      # box plot for pectral
       plt.figure(plt.gcf().number + 1)
       nrows, ncols = df_spec.shape
       df_spec.columns = ["PSR" + str(i+1) for i in range(ncols)]
       temp_plot = sns.boxplot(data = df_spec)

def ink_pattern_analysis(df_ink, folder_location, out_file_name):    # 잉크의 패턴검사
       nrows, ncols = df_ink.shape
       df_ink_clip = df_ink.clip(0,1)     # 최소 0 최대 1값으로 변경
       df_ink_clip = (df_ink_clip.astype(int)).astype(str)     # double->int->str
       patterns = []
       df_new = df_ink_clip.stack().groupby(level=0).apply(''.join).to_frame(0)
       patterns_dic = {}     # {pattern:count}
       for i in range(nrows):
              if df_new.iloc[i,0] in patterns_dic:    patterns_dic[df_new.iloc[i,0]] += 1
              else:patterns_dic.update({df_new.iloc[i,0]:1})
       patterns_count_list = sorted(patterns_dic.items(), key=lambda x: x[1])     # 내림차순정렬
       patterns_count_list = [x[1] for x in patterns_count_list]

       # plt.figure(plt.gcf().number + 1)
       # temp_plot = sns.boxplot(data = patterns_count_list)

       analized_df = (pd.DataFrame(patterns_count_list, columns = ['occurence of patterns']))
       fundamental_analysis(analized_df, folder_location, "out_patterns_fundamental_analysis_results.csv")
       (pd.DataFrame(patterns_dic, index = [0])).transpose().to_csv(folder_location + '/' + out_file_name, sep=',', index = False)       # 기초 통계 결과출력


def reducing_skewness(df_spectral):
       nrows, ncols = df_spectral.shape
       for i in range(ncols):
              if df_spectral.iloc[:,i].skew() > -1 and df_spectral.iloc[:,i].skew() < 1:
                     continue
              
              print(f'skewness, before = {df_spectral.iloc[:,i].skew():4f}')
              if plot_tag:
                     bins = np.arange(0,100,1)
                     plt.figure(plt.gcf().number)
                     plt.subplot(1,2,1)
                     plt.hist(df_spectral, bins, rwidth = 0.8)
                     plt.xlabel('reflection(%)')
                     plt.title('histogram of reflections')
              
              if df_spectral.iloc[:,i].skew() < -1:
                     maxval = df_spectral.iloc[:,i].max()
                     df_spectral.iloc[:,i] = maxval + 1.0 - df_spectral.iloc[:,i].astype(float)   # reflection
              df_spectral.iloc[:,i] = pd.DataFrame(np.log(df_spectral.iloc[:,i]))   # logarithm

              if plot_tag:
                     plt.subplot(1,2,2)
                     minval, maxval = df_spectral.iloc[:,i].min(), df_spectral.iloc[:,i].max()
                     bins = np.arange(minval, maxval, 0.1)
                     plt.hist(df_spectral, bins, rwidth = 0.8)
                     plt.xlabel('reflection(%)')
                     plt.title('histogram of reflections (reducing skewness)')
              print(f'skewness, after = {df_spectral.iloc[:,i].skew():4f}')


def PCA_analysis(df):
       nrows, ncols = df.shape
       pca = PCA(n_components=ncols)
       # pca.fit(df)
       score = pca.fit_transform(df)
       print(pca.explained_variance_ratio_)
       # plt.figure(plt.gcf().number)
       # plt.scatter(score[:,0], score[:,1])
       # print(pca.components_[0,:])
       # print('pca mean')
       # print(pca.mean_)
       # plt.figure(plt.gcf().number)
       # plt.plot(pca.components_[0,:])
       # plt.figure(plt.gcf().number +1)
       # plt.plot(pca.mean_)

def min_max_normalization(df):
       
       scaler = preprocessing.MinMaxScaler()
       df = scaler.fit_transform(df, axis = 0)
       return df

def data_normalization(df, paper_size, paper_spectral, inks_combination, measure_spectral, combi_type):

       scaler = preprocessing.StandardScaler()
       df_spec_paper = df.iloc[:, paper_spectral]  # paper spectral
       df_spec_all = df.iloc[:, measure_spectral]  # all spectral
       df_paper_wdw = df.iloc[:, paper_size]
       df_inks = df.iloc[:, inks_combination]  # inks data

       df_spec_paper_normal = pd.DataFrame(scaler.fit_transform(df_spec_paper), index=df_spec_paper.index, columns=df_spec_paper.columns)
       df_spec_all_normal = pd.DataFrame(scaler.fit_transform(df_spec_all), index=df_spec_all.index, columns=df_spec_all.columns)
       df_paper_wdw_normal = pd.DataFrame(scaler.fit_transform(df_paper_wdw), index=df_paper_wdw.index, columns=df_paper_wdw.columns)
       df_inks_normal = pd.DataFrame(df_inks.div(df_inks.sum(axis=1), axis=0), index=df_inks.index, columns=df_inks.columns)

       df_combine = pd.DataFrame()
       df_combine = pd.concat([df_paper_wdw_normal, df_inks_normal], axis=1)

       df_norm = deepcopy(df)
       # df_norm.loc[:, 'PAPER_SPECR_01':'PAPER_SPECR_31'] = df_spec_paper_normal
       # df_norm.loc[:, ['WIDTH', 'HEIGHT' ,'WEIGHT']] = df_paper_wdw_normal
       # df_norm.iloc[:, 70:125] = df_inks_normal

       df_norm.iloc[:, paper_spectral] = df_spec_paper_normal
       df_norm.iloc[:, paper_size] = df_paper_wdw_normal
       df_norm.iloc[:, inks_combination] = df_inks_normal


       return df_norm, df_combine
       # df.loc[:, 'SPECR_01':'SPECR_31'] = 

def data_analysis(df, folder_location):
       # df.columns = df.columns.astype(str)
       df_spec_paper = df.loc[:, 'PAPER_SPECR_01':'PAPER_SPECR_31']  # paper spectral
       df_spec_all = df.loc[:, 'SPECR_01':'SPECR_31']  # all spectral
       df_paper_wdw = df.loc[:, ['WIDTH', 'HEIGHT' ,'WEIGHT']]
       df_inks = df.iloc[:, 70:125]  # inks data

       PCA_analysis(df_spec_paper)
       PCA_analysis(df_spec_all)
       PCA_analysis(df_paper_wdw)
       PCA_analysis(df_inks)

       # df_spec_paper_integrated = pd.DataFrame(pd.Series(df_spec_paper.values.ravel('C')))
       # print(type(df_spec_paper_integrated))
       # reducing_skewness(df_spec_paper_integrated)
       # print(type(df_spec_paper_integrated))
       # print(df_spec_paper_integrated.mean()[1])
       # print(f'mean = {df_spec_paper_integrated.mean():.4f}, std = {df_spec_paper_integrated.std():.4f}')
       # df_spec_paper_reducing_skewness = reducing_skewness(df_spec_paper)
       # show_box_plot_spectral(df_spec_paper_reducing_skewness)
       
       # df_spec_all_reducing_skewness = normalize_spectral_data(df_spec_all)
       # show_box_plot_spectral(df_spec_all_reducing_skewness)
       # fundamental_analysis(df, folder_location, "out_analysis_results.csv")
       # if print_tag: print("1. fundamental analysis write done")
       # show_box_plot_paper(df)
       # if print_tag: print("2. box plot for the papers spectral done")
       # show_box_plot_spectral(df_spec_paper)
       # # show_box_plot_spectral(df_spec_paper_log)

       # if print_tag: print("3. box plots for the paper spectral done")
       # show_box_plot_spectral(df_spec_all)
       # # show_box_plot_spectral(df_spec_all_log)

       # if print_tag: print("4. box plots df_spec_allfor the all spectral done")
       ink_pattern_analysis(df_inks, folder_location, "out_pattern_analysis_results.csv")
       # if print_tag: print("5. box plots for the patterns dones")
       plt.show()