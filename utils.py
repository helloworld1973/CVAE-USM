import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def get_DDA_TRA_data(S_x, S_y, T_x, T_y, batch_size, num_D, width, device):
    S_x_func = torch.tensor(S_x).to(torch.float32)
    S_x_func = S_x_func.view(-1, num_D, 1, width).to(device)
    S_y_func = torch.tensor(S_y)
    S_d = [0 for i in range(len(S_y_func))]
    S_d_func = torch.tensor(S_d).to(device)
    S_idx = torch.arange(len(S_y_func)).to(device)



    T_x_func = torch.tensor(T_x).to(torch.float32)
    T_x_func = T_x_func.view(-1, num_D, 1, width).to(device)
    T_y_func = torch.tensor(T_y)
    T_d = [1 for i in range(len(T_y_func))]
    T_d_func = torch.tensor(T_d).to(device)
    T_idx = torch.arange(len(S_y_func), len(S_y_func) + len(T_y_func)).to(device)



    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x).to(torch.float32)
    ST_x_func = ST_x_func.view(-1, num_D, 1, width).to(device)
    ST_y_func = np.concatenate((S_y_func, T_y_func))
    ST_y_func = torch.tensor(ST_y_func)
    ST_d = S_d + T_d
    ST_d_func = torch.tensor(ST_d).to(device)
    ST_idx = torch.arange(len(ST_y_func)).to(device)
    ST_y_func = ST_y_func.to(device)
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_d_func, ST_idx)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    T_y_func = T_y_func.to(device)
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_d_func, T_idx)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    S_y_func = S_y_func.to(device)
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_d_func, S_idx)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader
