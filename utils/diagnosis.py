import pathlib
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

#---
# タイ順位に考慮した順位の計算
#---

def mean_rank(sorted_x, argsort_x):
    latest_value = sorted_x[0]
    latest_rank = 1
    same_value_rank_list = [argsort_x[0]] # タイ順位のリスト
    mean_rank_x = -np.ones_like(sorted_x).astype(np.float32)
    for i in range(1,len(sorted_x)):
        x = sorted_x[i]
        if x != latest_value:
            mean = (latest_rank + i) / 2 if i != 1 else 1
            mean_rank_x[same_value_rank_list] = mean
            latest_value = x
            latest_rank = i + 1
            same_value_rank_list = [argsort_x[i]]
        else:
            same_value_rank_list.append(argsort_x[i])
    mean = (latest_rank + len(sorted_x)) / 2
    mean_rank_x[same_value_rank_list] = mean
    
    return mean_rank_x

#---
# 順位相関の計算
#---

def calc_spearman_corr(x, y):
    
    #---
    # 配列を降順にソート
    #---
    
    sorted_x = -np.sort(-x)
    # 降順にするために-xをソートする
    sorted_y = -np.sort(-y)
    argsort_x = np.argsort(-x)
    argsort_y = np.argsort(-y)
    
    #---
    # 順位を算出
    #---
    
    rank_x = mean_rank(sorted_x, argsort_x)
    rank_y = mean_rank(sorted_y, argsort_y)
    
    #---
    # 順位相関係数を計算
    #---
    
    mean_x = np.mean(rank_x)
    mean_y = np.mean(rank_y)
    mean_xx = np.mean(rank_x*rank_x)
    mean_yy = np.mean(rank_y*rank_y)
    mean_xy = np.mean(rank_x*rank_y)
    corr = (mean_xy - mean_x*mean_y) / \
           (np.sqrt(mean_xx - mean_x**2) * np.sqrt(mean_yy - mean_y**2))
    
    return corr