import torch
import numpy as np
from gtda.time_series import SlidingWindow
from DDA_TRA.train import DDA_TRA_train
from DDA_TRA.util import log_and_print, matrix_to_string
from utils import get_DDA_TRA_data
import math

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
# PAMAP2_dataset
activity_list = ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs', 'descending_stairs',
                 'standing_in_an_elevator_still', 'moving_around_in_an_elevator',
                 'walking_in_a_parking_lot', 'walking_on_a_treadmill_in_flat',
                 'walking_on_a_treadmill_inclined_positions',
                 'running_on_a_treadmill_in_flat', 'exercising on a stepper', 'exercising on a cross trainer',
                 'cycling_on_an_exercise_bike_in_horizontal_positions',
                 'cycling_on_an_exercise_bike_in_vertical_positions',
                 'rowing', 'jumping', 'playing_basketball']
activities_required = activity_list
sensor_channels_required = ['RA_x_acc', 'RA_y_acc', 'RA_z_acc',
                            'RA_x_gyro', 'RA_y_gyro', 'RA_z_gyro']
source_user = '4'  # 2,4,7
target_user = '7'
Sampling_frequency = 25  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'DSADS'

def sliding_window_seg(data_x, data_y):
    # same setting as M1, except for no feature extraction step
    sliding_bag = SlidingWindow(size=int(Sampling_frequency * Num_Seconds),
                                stride=int(Sampling_frequency * Num_Seconds * (1 - Window_Overlap_Rate)))
    X_bags = sliding_bag.fit_transform(data_x)
    Y_bags = sliding_bag.resample(data_y)  # last occur label
    Y_bags = Y_bags.tolist()

    return X_bags, Y_bags

S_data = []
S_label = []
T_data = []
T_label = []

for index, a_act in enumerate(activities_required):
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_X_features.npy', 'rb') as f:
        source_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
        source_labels = np.load(f)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_X_features.npy', 'rb') as f:
        target_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
        target_labels = np.load(f)

    s_X_bags, s_Y_bags = sliding_window_seg(source_bags, source_labels)
    t_X_bags, t_Y_bags = sliding_window_seg(target_bags, target_labels)

    if index == 0:
        S_data = s_X_bags
        S_label = s_Y_bags
        T_data = t_X_bags
        T_label = t_Y_bags
    else:
        S_data = np.vstack((S_data, s_X_bags))
        S_label = S_label + s_Y_bags
        T_data = np.vstack((T_data, t_X_bags))
        T_label = T_label + t_Y_bags

S_label = [int(x) for x in S_label]
T_label = [int(x) for x in T_label]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model training paras settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:1")
lr_decay = 1.0
lr = 1e-2
num_D = 6
width = int(Sampling_frequency * Num_Seconds)
Num_classes = 19
global_epoch = 200
local_epoch_common = 1
local_epoch_RNN = 1
local_epoch_temporal = 1
time_lag_value = 3
print()
conv1_in_channels = num_D
conv1_out_channels = 16
conv2_out_channels = 32
full_connect_num = 100
kernel_size = 9
In_features_size = conv2_out_channels * math.floor(
    ((Num_Seconds * Sampling_frequency - kernel_size + 1) / 2 - kernel_size + 1) / 2)
GRL_alpha = 0.1
optim_Adam_weight_decay = 5e-4
optim_Adam_beta = 0.5
device = DEVICE
file_name = 'M4_DDA_TRA_' + str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_output.txt'
file_name_summary = 'M4_DDA_TRA_' + str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_output____________summary.txt'
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for local_epoch_RNN in [0, 2, 5, 10]:
    for GRL_alpha in [0.1, 0.2, 0.3]:
        for full_connect_num in [100, 80, 50]:
            for lr in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
                for time_lag_value in [3, 6, 9, 12]:
                    local_epoch_temporal = local_epoch_RNN
                    if local_epoch_RNN == 0:
                        local_epoch_common = 1
                    else:
                        local_epoch_common = local_epoch_RNN
                    print('para_setting:' + str(local_epoch_common) + '_' + str(
                        local_epoch_RNN) + '_' + str(local_epoch_temporal) + '_' + str(
                        GRL_alpha) + '_' + str(full_connect_num) + '_' + str(
                        lr) + '_' + str(time_lag_value))
                    log_and_print(content='para_setting:' + str(local_epoch_common) + '_' + str(
                        local_epoch_RNN) + '_' + str(local_epoch_temporal) + '_' + str(
                        GRL_alpha) + '_' + str(full_connect_num) + '_' + str(
                        lr) + '_' + str(time_lag_value), filename=file_name)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    S_torch_loader, T_torch_loader, ST_torch_loader = get_DDA_TRA_data(S_data, S_label, T_data, T_label,
                                                                                       batch_size=10000, num_D=num_D,
                                                                                       width=width, device=device)

                    target_acc = DDA_TRA_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch, local_epoch_common,
                                               local_epoch_RNN,
                                               local_epoch_temporal, time_lag_value,
                                               conv1_in_channels, conv1_out_channels, conv2_out_channels,
                                               full_connect_num, Num_classes, kernel_size, In_features_size, GRL_alpha,
                                               lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta, file_name, device)
                    best_target_acc, best_target_cm, corresponding_best_source_acc = DDA_TRA_train(S_torch_loader, T_torch_loader,
                                                                    ST_torch_loader, global_epoch,
                                                                    local_epoch_common,
                                                                    local_epoch_RNN,
                                                                    local_epoch_temporal, time_lag_value,
                                                                    conv1_in_channels, conv1_out_channels,
                                                                    conv2_out_channels,
                                                                    full_connect_num, Num_classes, kernel_size,
                                                                    In_features_size, GRL_alpha,
                                                                    lr_decay, lr, optim_Adam_weight_decay,
                                                                    optim_Adam_beta, file_name, device)
                    print()
                    log_and_print(
                        content='para_setting:' + str(local_epoch_common) + '_' + str(
                            local_epoch_RNN) + '_' + str(local_epoch_temporal) + '_' + str(
                            GRL_alpha) + '_' + str(full_connect_num) + '_' + str(
                            lr) + '_' + str(time_lag_value), filename=file_name_summary)
                    log_and_print(
                        content='best target acc:' + str(best_target_acc),
                        filename=file_name_summary)
                    log_and_print(
                        content='corresponding best source acc:' + str(corresponding_best_source_acc),
                        filename=file_name_summary)
                    log_and_print(
                        content='best cm:',
                        filename=file_name_summary)
                    log_and_print(
                        content=matrix_to_string(best_target_cm),
                        filename=file_name_summary)
                    log_and_print(
                        content='-------------------------------------------------',
                        filename=file_name_summary)
