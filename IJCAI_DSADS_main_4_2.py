import numpy as np
import torch
from gtda.time_series import SlidingWindow
import random
from IJCAI_CVAE_USM.train import GPU_CVAE_USM_train
import math

from IJCAI_CVAE_USM.utils.util import log_and_print, GPU_get_CVAE_USM_train_data, matrix_to_string

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
target_user = '2'
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


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model training paras settings
num_D = 6
width = Sampling_frequency * Num_Seconds
Num_classes = 19
Epochs = 400
Local_epoch = 1
device = torch.device("cuda:1")  # "cuda:2"
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# DGTSDA_temporal_diff model
Conv1_in_channels = num_D
Conv1_out_channels = 16
Conv2_out_channels = 32
Kernel_size_num = 9
In_features_size = Conv2_out_channels * math.floor(
    ((Num_Seconds * Sampling_frequency - Kernel_size_num + 1) / 2 - Kernel_size_num + 1) / 2)

Lr_decay = 1.0
Optim_Adam_weight_decay = 5e-4
Optim_Adam_beta = 0.5

Alpha = 5.0  # RECON_L
Beta = 10.0  # KLD_L
Delta = 5.0  # DOMAIN_L
Gamma = 60.0  # CLASS_L # 30
Epsilon = 60.0  # TEMPORAL_L
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
file_name = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_CVAE_USM.txt'
file_name_summary = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(
    target_user) + '_CVAE_USM_summary.txt'

for Lr_decay in [1.0]:  # 1.0, 0.8, 0.5
    for Optim_Adam_weight_decay in [5e-4]:  # 5e-4, 5e-3, 5e-2, 5e-1
        for Optim_Adam_beta in [0.2]:  # 0.2, 0.5, 0.9
            for Hidden_size in [100, 80, 50]:  # 100, 80, 50
                for Dis_hidden in [50, 30, 20]:  # 50, 30, 20
                    for ReverseLayer_latent_domain_alpha in [0.05, 0.1, 0.15, 0.2, 0.25,
                                                             0.3]:  # 0.2, 0.15, 0.25, 0.3, 0.1, 0.35
                        for lr in [5 * 1e-4, 1e-4, 5 * 1e-5, 1e-5]:  # 1e-2, 1e-1, 1e-3,1e-4, 1e-5, 1e-6
                            for Variance in [1, 2, 3, 4, 5]:  # 1, 2, 0.7, 3, 0.4, 4, 5
                                for num_sub_act in [2, 5, 10, 15, 20, 25, 30, 35]:
                                    for Num_temporal_states in [num_sub_act,
                                                                math.floor(num_sub_act * 1.5),
                                                                math.floor(num_sub_act * 2),
                                                                math.floor(num_sub_act * 2.5)]:  # 2, 3, 4, 5, 6, 7
                                        print('para_setting:' + str(num_sub_act) + '_' + str(
                                            Num_temporal_states) + '_' + str(
                                            Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                            Lr_decay) + '_' + str(
                                            Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                            Variance) + '_' + str(lr) + '_' + str(
                                            ReverseLayer_latent_domain_alpha))
                                        log_and_print(
                                            content='para_setting:' + str(num_sub_act) + '_' + str(
                                                Num_temporal_states) + '_' + str(
                                                Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                                Lr_decay) + '_' + str(
                                                Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                                Variance) + '_' + str(lr) + '_' + str(
                                                ReverseLayer_latent_domain_alpha), filename=file_name)

                                        S_torch_loader, T_torch_loader, ST_torch_loader = GPU_get_CVAE_USM_train_data(
                                            S_data, S_label, T_data, T_label,
                                            batch_size=10000, num_D=num_D,
                                            width=width,
                                            num_class=Num_classes, device=device)

                                        best_target_acc, best_target_cm, corresponding_best_source_acc, best_epoch = GPU_CVAE_USM_train(
                                            S_torch_loader,
                                            T_torch_loader,
                                            ST_torch_loader,
                                            global_epoch=Epochs,
                                            local_epoch=Local_epoch,
                                            num_classes=Num_classes,
                                            num_sub_act=num_sub_act,
                                            num_temporal_states=Num_temporal_states,
                                            conv1_in_channels=Conv1_in_channels,
                                            conv1_out_channels=Conv1_out_channels,
                                            conv2_out_channels=Conv2_out_channels,
                                            kernel_size_num=Kernel_size_num,
                                            in_features_size=In_features_size,
                                            hidden_size=Hidden_size,
                                            dis_hidden=Dis_hidden,
                                            ReverseLayer_latent_domain_alpha=ReverseLayer_latent_domain_alpha,
                                            variance=Variance,
                                            alpha=Alpha,
                                            beta=Beta,
                                            gamma=Gamma,
                                            delta=Delta,
                                            epsilon=Epsilon,
                                            lr_decay=Lr_decay,
                                            lr=lr,
                                            optim_Adam_weight_decay=Optim_Adam_weight_decay,
                                            optim_Adam_beta=Optim_Adam_beta,
                                            file_name=file_name,
                                            device=device)

                                        print()
                                        log_and_print(
                                            content='para_setting:' + str(num_sub_act) + '_' + str(
                                                Num_temporal_states) + '_' + str(
                                                Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                                Lr_decay) + '_' + str(
                                                Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                                Variance) + '_' + str(lr) + '_' + str(
                                                ReverseLayer_latent_domain_alpha), filename=file_name_summary)
                                        log_and_print(
                                            content='best target acc:' + str(best_target_acc),
                                            filename=file_name_summary)
                                        log_and_print(
                                            content='corresponding best source acc:' + str(
                                                corresponding_best_source_acc),
                                            filename=file_name_summary)
                                        log_and_print(
                                            content='best cm:',
                                            filename=file_name_summary)
                                        log_and_print(
                                            content=matrix_to_string(best_target_cm),
                                            filename=file_name_summary)
                                        log_and_print(
                                            content='best epoch:' + str(best_epoch),
                                            filename=file_name_summary)
                                        log_and_print(
                                            content='-------------------------------------------------',
                                            filename=file_name_summary)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
