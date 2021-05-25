import matplotlib.pyplot as plt
from os import makedirs
import numpy as np
import pandas as pd
import csv
from os import path


def plot_loss_trends(train_loss, valid_loss):

    # 훈련이 진행되는 과정에 따라 loss를 시각화
    # fig = plt.figure(figsize=(10,8))
    fig1 = plt.figure(plt.gcf().number)
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # validation loss의 최저값 지점을 찾기
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5) # 일정한 scale
    # plt.xlim(0, len(train_loss)+1) # 일정한 scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig1.savefig('loss_plot.png', bbox_inches = 'tight')

def plot_spectral_results(Y_input, df_result_test_losses, X_input, df_result_prediction, plot_path):

    spec_len = 31
    x_axis = pd.DataFrame([i+1 for i in range(spec_len)])
    # test_idxs = pd.DataFrame(test_idxs)

    test_idxs = df_result_prediction.index
    # df_result_ = pd.DataFrame(df_result_prediction)
    # df_result_.insert(0, "idx", test_idxs)
    # df_result_test_losses = pd.DataFrame(df_result_test_losses, columns=['loss'])
    # df_result_test_losses.insert(0, 'idx', test_idxs)
    df_result_test_losses = df_result_test_losses.sort_values(by=['loss'], ascending=False) # MSE 제일 큰 순서로 sort
    # df_result_test_losses.index = range(len(df_result_test_losses))   # re indexing
    
    # x_test_sorted_by_loss = X_input.iloc[X_input.index.get_indexer(df_result_test_losses.index)]
    # y_test = Y_input.iloc[Y_input.index.get_indexer(df_result_test_losses['idx'])]

    # df_result_
    y_predicted_avg = pd.DataFrame()     # output spectral reflectance
    x_avg = pd.DataFrame() # input paper spectral reflectance   (x)
    y_avg = pd.DataFrame()  # y
    fig1 = plt.figure(plt.gcf().number)
    subplot_shape = (4,3)
    text_position = (3, 80)
    # for i in range(len(x_test_sorted_by_loss)):    # PLOT FIGURES BY MSE VALUES 
    for i in range(len(df_result_test_losses)):    # PLOT FIGURES BY MSE VALUES 
        idx = df_result_test_losses.index[i]
        # x = pd.DataFrame(np.array(X_input.loc[idx, X_input.columns.slice_indexer('PAPER_SPECR_01','PAPER_SPECR_31')]).reshape(1,-1)) # spectral input (x)
        x = pd.DataFrame(np.array(X_input.loc[idx, 'PAPER_SPECR_01':'PAPER_SPECR_31']).reshape(1,-1))
        x_avg = pd.concat([x_avg, x])
        y_predicted = pd.DataFrame(np.array(df_result_prediction.loc[idx]).reshape(1,-1))  # spectral output (y_predicted) < - 인덱스별로해야함
        # y = pd.DataFrame(np.array(y_test.iloc[i,:]).reshape(1,-1))  # y
        y = pd.DataFrame(np.array(Y_input.loc[idx]).reshape(1,-1))  # y
        y_avg = pd.concat([y_avg, y])
        y_predicted_avg = pd.concat([y_predicted_avg, y_predicted])

        if i > subplot_shape[0] * subplot_shape[1] - 1: # PLOT TOP 12 MSE VALUES
            continue
        plt.subplot(subplot_shape[0], subplot_shape[1], i+1)    # subplot
        plt.plot(np.array(x_axis), np.array(x.T)) # plot input
        plt.plot(np.array(x_axis), np.array(y.T))  # plot y
        plt.plot(np.array(x_axis), np.array(y_predicted.T)) # plot y_predicted
        plt.xlabel('wavelengh')
        plt.ylabel('reflection(%)')
        plt.legend(['x','y', 'y_predicted'])
        # plt.legend(['y', 'y_predicted'])
        plt.axis([0,31,0,100])
        # plt.text(3,80,'data idx = {0}, MSE = {1:.2f}'.format(x_test_sorted_by_loss.index[i],df_result_test_losses['loss'][i]))
        plt.text(3,80,'data idx = {0}, MSE = {1:.2f}'.format(idx, df_result_test_losses.iloc[i][0]))
    makedirs(plot_path, exist_ok=True)
    fig1.savefig('spectral_results/spectral_result{0}.png'.format(i), bbox_inches = 'tight')

    y_predicted_avg = y_predicted_avg.mean()
    y_avg = y_avg.mean()
    x_avg = x_avg.mean()

    fig2 = plt.figure(plt.gcf().number + 1)
    # df_result_test_losses = np.array(df_result_test_losses)
    # test_losses_avg = df_result_test_losses.mean()
    test_losses_avg = df_result_test_losses['loss'].mean()
    # plt.plot(np.array(x_axis), np.array(x_avg))    # plot x avg
    plt.plot(np.array(x_axis), np.array(y_avg))  # plot y avg
    plt.plot(np.array(x_axis), np.array(y_predicted_avg))    # plot y_predicted avg
    plt.title('x vs. y vs. y_predicted (average)')
    # plt.title('y vs. y_predicted (average)')
    plt.xlabel('wavelengh')
    plt.ylabel('reflection(%)')
    plt.legend(['x', 'y', 'y_predicted'])
    # plt.legend(['y', 'y_predicted'])
    plt.text(3,80,'avg. MSE = {0:.2f}'.format(test_losses_avg))
    plt.axis([0,31,0,100])
    fig2.savefig('spectral_results/spectral_result_average.png', bbox_inches = 'tight')

def write_spectral_result(Y_input, test_idxs, directory, filename):
    if not path.exists(directory):
        path.makedirs(directory)
    csv_file_name = directory + filename + '.csv'
    f = open(csv_file_name , 'a', newline='')
    fwriter = csv.writer(f)


    y_test = Y_input.iloc[Y_input.index.get_indexer(test_idxs)]


def write_spectral_result_detail(Y_input, df_result_prediction, df_result_test_losses, directory, filename):

    df_result_prediction.sort_index()
    df_result_test_losses.sort_index()
    df_result_prediction.insert(-1, df_result_test_losses.columns, df_result_test_losses[df_result_test_losses.columns])
    df_result_prediction.to_csv(directory + '/' + filename, sep=',', index = True)       # 기초 통계 결과출력



def write_all_results(train_loss, valid_loss, Y_input, df_result_test_losses, X_input_original, df_result_prediction, spect_plot_path):
    
    plot_loss_trends(train_loss, valid_loss)
    plot_spectral_results(Y_input, df_result_test_losses, X_input_original, df_result_prediction, spect_plot_path)
    plt.show()



