import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def get_TrC_Target_user_data(x, y, batch_size, num_D, width):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42, stratify=y)

    X_test = torch.tensor(X_test).to(torch.float32)
    X_test = X_test.view(-1, num_D, 1, width)
    y_test = torch.tensor(y_test)
    small_amount_torch_dataset = Data.TensorDataset(X_test, y_test)
    T_part_torch_loader = Data.DataLoader(dataset=small_amount_torch_dataset, batch_size=batch_size, shuffle=False)


    x = torch.tensor(x).to(torch.float32)
    x = x.view(-1, num_D, 1, width)
    y = torch.tensor(y)
    torch_dataset = Data.TensorDataset(x, y)
    T_all_torch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
    return T_part_torch_loader, T_all_torch_loader


def get_TrC_Source_user_data(x, y, batch_size, num_D, width):
    x = torch.tensor(x).to(torch.float32)
    x = x.view(-1, num_D, 1, width)
    y = torch.tensor(y)
    torch_dataset = Data.TensorDataset(x, y)
    torch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
    return torch_loader


def get_DANN_data(x, y, batch_size, num_D, width):
    x = torch.tensor(x).to(torch.float32)
    x = x.view(-1, num_D, 1, width)
    y = torch.tensor(y)
    torch_dataset = Data.TensorDataset(x, y)
    torch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
    return torch_loader


def get_DIVERSITY_data(x, y, batch_size, num_D, width):
    x = torch.tensor(x).to(torch.float32)
    x = x.view(-1, num_D, 1, width)
    y = torch.tensor(y)
    predict_d = torch.zeros(len(y))
    idx = torch.arange(len(y))
    torch_dataset = Data.TensorDataset(x, y, predict_d, idx)
    torch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
    return torch_loader


def get_DGTSDA_temporal_diff_train_data(S_x, S_y, T_x, T_y, batch_size, num_D, width, num_class):
    S_x_func = torch.tensor(S_x).to(torch.float32)
    S_x_func = S_x_func.view(-1, num_D, 1, width)
    S_y_func = torch.tensor(S_y)
    S_predict_ts = torch.zeros(len(S_y_func))
    S_d = [0 for i in range(len(S_y_func))]
    S_d_func = torch.tensor(S_d)
    S_idx = torch.arange(len(S_y_func))
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_predict_ts, S_d_func, S_idx)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    T_x_func = torch.tensor(T_x).to(torch.float32)
    T_x_func = T_x_func.view(-1, num_D, 1, width)
    T_y_func = torch.tensor(T_y)
    T_predict_ts = torch.zeros(len(T_y_func))
    T_d = [1 for i in range(len(T_y_func))]
    T_d_func = torch.tensor(T_d)
    T_idx = torch.arange(len(S_y_func), len(S_y_func) + len(T_y_func))
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_predict_ts, T_d_func, T_idx)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x).to(torch.float32)
    ST_x_func = ST_x_func.view(-1, num_D, 1, width)
    T_y_new = [i + num_class for i in T_y]
    ST_y = S_y + T_y_new
    ST_y_func = torch.tensor(ST_y)
    ST_predict_ts = torch.zeros(len(ST_y_func))
    ST_d = S_d + T_d
    ST_d_func = torch.tensor(ST_d)
    ST_idx = torch.arange(len(ST_y_func))
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_predict_ts, ST_d_func, ST_idx)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader
