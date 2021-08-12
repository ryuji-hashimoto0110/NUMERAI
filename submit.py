import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

def submit(train_models, test_Xs, test_ts, dataset_class, test_id, 
           device, transforms,
           root_path, folder_name='numerai_datasets'):
    for model in train_models:
        model.eval()
        
    model_num = len(train_models)
    dataset_test = dataset_class(test_Xs, test_ts, transforms)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=1000, shuffle=False)
    preds = []
    correct = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader_test):
            bs = batch[0].shape[0]
            xs = batch[0].to(device)
            _preds = np.zeros(bs).astype(np.float32)
            for model in train_models:
                model_preds = model(xs).detach().cpu().numpy()
                _preds += model_preds / model_num
            preds.extend(_preds.tolist())
    
    preds_df = pd.DataFrame({'id': test_id,
                             'prediction': preds})
    preds_csv_path = root_path / folder_name / 'preds.csv'
    preds_df.to_csv(preds_csv_path, index=False)
