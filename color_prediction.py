import data_analysis
import deep_learning
import pandas as pd
import numpy as np
import write_result
import deep_learning

resume = True # 저장점부터 재시작
training_model = False    # model training하면 True
# testing_model = True     # 주어진 모델 test

data_location = "./data_project"    
five_raw_data_names = ["log_spectrodetail.csv", "log_spectrok_pkpaper.csv", "mst_ink.csv", "mst_paper.csv", "udt_imxbom.csv"]
df_original = pd.read_csv(data_location + '/20210302_add_paper.csv')
df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed')]
df_original.columns = df_original.columns.astype(str)
df_deleted = data_analysis.delete_outlier(df_original, data_location, "out_deleted_data.csv", "out_preprocessed_data.csv")
df_deleted.insert(4, "P_DENSITY", df_deleted.WEIGHT/df_deleted.WIDTH*df_deleted.HEIGHT)   # paper density 계산해서 추가
# paper_density = df_deleted.columns.slice_indexer('WIDTH','WEIGHT')  # wdw only     (3)
paper_density = df_deleted.columns.slice_indexer('P_DENSITY', 'P_DENSITY')  # wdw only     (3)
paper_spectral = df_deleted.columns.slice_indexer('PAPER_SPECR_01','PAPER_SPECR_31')  # paper spectral only (31)
inks_combination = df_deleted.columns.slice_indexer('20120002','123')  # inks only (56)
measure_spectral = df_deleted.columns.slice_indexer('SPECR_01', 'SPECR_31')

combi_type = 1
df_new_norm, df_combine_normal = data_analysis.data_normalization(df_deleted, paper_density, paper_spectral, inks_combination, measure_spectral, combi_type)
df_new_norm = df_new_norm.dropna(subset=df_new_norm.columns[inks_combination]) # ink combination이 없는 missing value 제거
deep_learning.df_full_input = df_new_norm
Y_input = df_new_norm.iloc[:, measure_spectral]  # Y input
X_input_norm = df_new_norm.iloc[:, np.r_[paper_density, paper_spectral, inks_combination]]  # normalized X input
device = deep_learning.torch.device('cuda' if  deep_learning.torch.cuda.is_available() else 'cpu')
avg_train_losses = [] # epoch당 average training loss를 track
avg_valid_losses = [] # epoch당 average validation loss를 track

# Hyper-parameters
input_size = len(X_input_norm.columns) # n features
num_output = len(Y_input.columns)
num_epochs = 5000
batch_size = 32
patience = 2
# hidden_sizes = [300, 200]
hidden_sizes = [300, 200]
learning_rate = 0.001

# data set
model = deep_learning.NeuralNet(input_size, hidden_sizes, num_output).to(device)
train_valid_test_ratio = np.array([0.8, 0.07, 0.13])   # train / valid / test 비율 정의
x_train, x_valid, x_test, y_train, y_valid, y_test = deep_learning.get_train_valid_test(X_input_norm, Y_input, train_valid_test_ratio)
# # Loss and optimizer
criterion = deep_learning.nn.MSELoss()
optimizer = deep_learning.torch.optim.Adam(model.parameters(), lr=learning_rate)  
scheduler = deep_learning.torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda=lambda epoch:0.9**epoch)
start_epoch = 0

if resume == True:
     checkpoint = deep_learning.es.torch.load('./checkpoints/checkpoint.pt')  # save checkpoint
     model.load_state_dict(checkpoint['model_state_dict'])
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
     start_epoch = checkpoint['epoch']
     avg_train_losses = checkpoint['train_loss']
     avg_valid_losses = checkpoint['valid_loss']
     train_idxs = checkpoint['train_idxs']
     valid_idxs = checkpoint['valid_idxs']
     test_idxs = checkpoint['test_idxs']
     x_train = X_input_norm[X_input_norm.index.isin(train_idxs)]
     x_valid = X_input_norm[X_input_norm.index.isin(valid_idxs)]
     x_test = X_input_norm[X_input_norm.index.isin(test_idxs)]
     y_train = Y_input[Y_input.index.isin(train_idxs)]
     y_valid = Y_input[Y_input.index.isin(valid_idxs)]
     y_test = Y_input[Y_input.index.isin(test_idxs)]

dataset_train = deep_learning.MyDataset(x_train, y_train)
dataset_valid = deep_learning.MyDataset(x_valid, y_valid)
dataset_test = deep_learning.MyDataset(x_test, y_test)
train_loader = deep_learning.torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
valid_loader = deep_learning.torch.utils.data.DataLoader(dataset=dataset_valid, shuffle=True)
test_loader = deep_learning.torch.utils.data.DataLoader(dataset=dataset_test, shuffle=False)


######################
# TRAINING the model #
######################
if training_model:
     # model, train_loss, valid_loss, df_result_test_losses, df_result_prediction, train_idxs, valid_idxs\
     model, avg_train_losses, avg_valid_losses, early_stopping\
          = deep_learning.do_learning(model, optimizer, scheduler, device, start_epoch,
          num_epochs, batch_size, patience, avg_train_losses, avg_valid_losses,
         train_loader, valid_loader, criterion, x_train.index, x_valid.index, x_test.index, resume)
     # save({'train_idxs': x_train.index,'test_idxs': x_test.index,'valid_idxs': x_valid.index}, early_stopping.path)

######################
# TEST the model #
######################
test_losses, predicted_spectral_list = deep_learning.test_model(model, test_loader, criterion, device)
df_result_prediction = pd.DataFrame(predicted_spectral_list)
df_result_test_losses = pd.DataFrame(test_losses, columns=['loss'])
df_result_test_losses.index = df_result_prediction.index =  x_test.index
df_result_prediction.columns = df_deleted.columns[df_deleted.columns.slice_indexer("SPECR_01","SPECR_31")]   # copy column names


# input select
X_input_original = df_original.iloc[X_input_norm.index, np.r_[paper_density, paper_spectral, inks_combination]]  # original X input
# X_input_original = df_original.iloc[X_input_norm.index, np.r_[paper_density, paper_spectral]]  # original X input
# X_input_original = df_original.iloc[X_input_norm.index, np.r_[paper_density, inks_combination]]  # original X input
# X_input_original = df_original.iloc[X_input_norm.index, np.r_[paper_spectral, inks_combination]]  # original X input
spect_plot_path = './spectral_results'
write_result.write_all_results(avg_train_losses, avg_valid_losses, Y_input, df_result_test_losses, X_input_original, df_result_prediction, spect_plot_path)