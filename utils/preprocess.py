import pathlib
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

#---
# "numerai_datasets"フォルダからtrain/tournament用csvファイルを読み込み
#---

def make_traintest_data(root_path):
    numerai_dataset_path = root_path / 'numerai_datasets'
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

