import pathlib
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import tensorboardX as tbx
from utils.diagnosis import calc_spearman_corr, mean_rank

def train(model_class, dataset_class, model_name, checkpoints_path, log_dir,
          fold_n, num_epoch, input_dim, device, transforms,
          train_Xs, train_ts,
          criterion):
    kf = KFold(n_splits=fold_n, shuffle=True, random_state=123)
    spl = kf.split(train_Xs)
    train_models = []
    train_model_paths = []
    print(f'[device]{device}')
    
    #---
    # fold
    #---
    
    for fold_i, (train_idx, val_idx) in enumerate(spl):
        
        train_losses = []
        train_corres = []
        val_losses = []
        val_corres = []
        
        print(f'[fold]{fold_i+1}/{fold_n}')
        print(f'[train_n]{len(train_idx)} [val_n]{len(val_idx)}')
        
        #---
        # dataset
        #---
        
        X_train = train_Xs[train_idx]
        t_train = train_ts[train_idx]
        X_val = train_Xs[val_idx]
        t_val = train_ts[val_idx]
        dataset_train = dataset_class(X_train, t_train, transforms)
        dataset_val = dataset_class(X_val, t_val, transforms)
        train_n = len(X_train)
        val_n = len(X_val)
        
        #---
        # dataloader
        #---
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=1000, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=1000, shuffle=False)
        
        #---
        # model, criterion, optimizer
        #---
        
        model = model_class(input_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        #---
        # TensorBoard
        #---
        
        writer = tbx.SummaryWriter(log_dir)
        
        #---
        # epoch
        #---
        
        tr_losses = []
        tr_corres = []
        val_losses = []
        val_corres = []
        for epoch in range(num_epoch):
            
            #---
            # train
            #---
            
            model.train()
            tr_loss = 0
            preds = []
            labels = []
            train_time_start = time.time()
            for step, batch in enumerate(dataloader_train):
                optimizer.zero_grad()
                bs = batch[0].shape[0]
                xs = batch[0].to(device)
                ts = batch[1].view(bs,1).to(device)
                ys = model(xs)
                loss = criterion(ys, ts)
                loss.backward()
                optimizer.step()
                tr_loss += (loss.item() * bs) / train_n
                preds.extend(ys.detach().cpu().numpy().tolist())
                labels.extend(ts.detach().cpu().numpy().tolist())
            preds = np.array(preds).astype(np.float32)
            labels = np.array(labels).astype(np.float32)
            tr_corr = calc_spearman_corr(preds[:,0], labels[:,0])
            tr_losses.append(tr_loss)
            tr_corres.append(tr_corr)
            train_time_end = time.time()
            train_time_total = train_time_end - train_time_start
                
            #---
            # val
            #---
            
            model.eval()
            val_loss = 0
            preds = []
            labels = []
            val_time_start = time.time()
            with torch.no_grad():
                for step, batch in enumerate(dataloader_val):
                    bs = batch[0].shape[0]
                    xs = batch[0].to(device)
                    ts = batch[1].view(bs,1).to(device)
                    ys = model(xs)
                    loss = criterion(ys, ts)
                    val_loss += (loss.item() * bs) / val_n
                    preds.extend(ys.detach().cpu().numpy().tolist())
                    labels.extend(ts.detach().cpu().numpy().tolist())
            preds = np.array(preds).astype(np.float32)
            labels = np.array(labels).astype(np.float32)
            val_corr = calc_spearman_corr(preds[:,0], labels[:,0])
            val_losses.append(val_loss)
            val_corres.append(val_corr)
            writer.add_scalars(f'Loss/fold{fold_i+1}',
                          {
                              'train': tr_loss,
                              'val': val_loss
                          }, epoch)
            writer.add_scalars(f'Corr/fold{fold_i+1}',
                          {
                              'train': tr_corr,
                              'val': val_corr
                          }, epoch)
            val_time_end = time.time()
            val_time_total = val_time_end - val_time_start
            total_time = train_time_total + val_time_total
            
            print(f'[epoch]{epoch+1}/{num_epoch}' +
                  f' [loss]tra:{tr_loss:.4f} val:{val_loss:.4f}' + 
                  f' [corr]tra:{tr_corr:.8f} val:{val_corr:.8f}' + 
                  f' [time]total:{total_time:.2f}sec' +
                  f' tra:{train_time_total}sec' +
                  f' val:{val_time_total}sec')
            
        #---
        # save model
        #---
        
        savename = f'{model_name}_epoch{num_epoch}_fold{fold_i+1}.pth'
        save_path = checkpoints_path / savename
        if not checkpoints_path.exists():
            checkpoints_path.mkdir(parents=True)
        torch.save(model.state_dict(), save_path)
        print(f'model saved to >> {save_path}')
        print()
        train_models.append(model)
        train_model_paths.append(save_path)
        writer.close()
        
        #---
        # display loss and corr
        #---
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.grid()
        ax1.plot(tr_losses, marker='.', markersize=6, color='red', label='train loss')
        ax1.plot(val_losses, marker='.', markersize=6, color='blue', label='val loss')
        ax2.plot(tr_corres, marker='.', markersize=6, color='green', label='train corr')
        ax2.plot(val_corres, marker='.', markersize=6, color='orange', label='val corr')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc = 'upper right')
        ax1.set(xlabel = 'Epoch', ylabel = 'Loss')
        ax2.set(ylabel = 'Corr')
        plt.show()
        
    return train_models, train_model_paths