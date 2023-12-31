import random
import numpy as np
import torch
import torch.utils.data as Data


def GPU_get_CVAE_USM_train_data(S_x, S_y, T_x, T_y, batch_size, num_D, width, num_class, device):
    # Define the device

    # Process S data
    S_x_func = torch.tensor(S_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    S_y_func = torch.tensor(S_y).to(device)
    S_predict_ts = torch.zeros(len(S_y_func)).to(device)
    S_d_func = torch.zeros(len(S_y_func)).to(device)  # Simplified from a loop
    S_idx = torch.arange(len(S_y_func)).to(device)
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_predict_ts, S_d_func, S_idx)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process T data
    T_x_func = torch.tensor(T_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    T_y_func = torch.tensor(T_y).to(device)
    T_predict_ts = torch.zeros(len(T_y_func)).to(device)
    T_d_func = torch.ones(len(T_y_func)).to(device)  # Simplified from a loop
    T_idx = torch.arange(len(S_y_func), len(S_y_func) + len(T_y_func)).to(device)
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_predict_ts, T_d_func, T_idx)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process combined ST data
    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x, dtype=torch.float32).view(-1, num_D, 1, width).to(device)
    T_y_new = [i + num_class for i in T_y]
    ST_y = S_y + T_y_new
    ST_y_func = torch.tensor(ST_y).to(device)
    ST_predict_ts = torch.zeros(len(ST_y_func)).to(device)
    ST_d_func = torch.cat([S_d_func, T_d_func]).to(device)
    ST_idx = torch.arange(len(ST_y_func)).to(device)
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_predict_ts, ST_d_func, ST_idx)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader


def log_and_print(content, filename):
    """
    Prints the content to the console and also writes it to a file.

    Args:
    - content (str): The content to be printed and logged.
    - filename (str): The name of the file to which the content will be written.
    """
    with open(filename, 'a') as file:  # 'a' ensures content is appended and doesn't overwrite existing content
        file.write(content + '\n')
    # print(content)


def print_row(row, colwidth=10, latex=False, file_name=''):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)
    Content = sep.join([format_val(x) for x in row])
    log_and_print(content=Content, filename=file_name)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def matrix_to_string(matrix):
    return '\n'.join(['\t'.join(map(str, row)) for row in matrix])
