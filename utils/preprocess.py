import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils.diagnosis import mean_rank, calc_spearman_corr

#---
# "numerai_datasets"フォルダからtrain/tournament用csvファイルを読み込み
#---

def make_traintest_data(root_path, folder_name='numerai_datasets'):
    numerai_dataset_path = root_path / folder_name
    train_data_path = numerai_dataset_path / 'numerai_training_data.csv'
    test_data_path = numerai_dataset_path / 'numerai_tournament_data.csv'
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    feature_names = [f for f in train_data.columns if "feature" in f]
    return train_data, test_data, feature_names

#---
# era毎にサンプルサイズを描画
#---

def count_samplesize_by_era(df, figsize=(15,5)):
    samplesize_series = df['era'].value_counts() 
    # 'era'列についてユニークな要素それぞれの出現回数を算出．
    # 参考：https://note.nkmk.me/python-pandas-value-counts/
    era_list = []
    samplesize_list = []
    
    # era_list=[1,2,3,...,120], samplesilze_list=[samplesize_series['era1'],...]とする．
    for i in range(1,121):
        era = f'era{i}'
        era_list.append(i)
        samplesize_list.append(samplesize_series[era])
    
    plt.figure(figsize=figsize)
    plt.bar(x=era_list, height=samplesize_list)
    plt.xlabel('era')
    plt.ylabel('sample size')
    plt.show()
    
    return samplesize_list


#---
# targetとの順位相関が閾値よりも大きい特徴量を抽出
#---

def choose_features(df, feature_names, threshold=0.005):
    highcorr_features = []
    target = df['target'].values
    features = df[feature_names].values
    corr_arr = np.apply_along_axis(calc_spearman_corr, axis=0, 
                                   arr=features, y=target)
    highcorr_idx = np.where(corr_arr > threshold)
    highcorr_arr = corr_arr[highcorr_idx]
    return highcorr_idx, highcorr_arr

#---
# テーブル形式のデータセットutils
#---

class dfDataset(Dataset):
    def __init__(self, Xs, ts):
        self.Xs = Xs
        self.ts = ts
        
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.xs[index,:].astype(np.float32))
        t = torch.from_numpy(self.ts[index,:].astype(np.float32))
        return x, t

