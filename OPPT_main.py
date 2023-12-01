import torch
import numpy as np
from gtda.time_series import SlidingWindow
import random
from utils import get_DANN_data
from DANN.train import DANN_train

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm
activity_list = ['Stand', 'Walk', 'Sit', 'Lie']
DATASET_NAME = 'OPPT'
activities_required = activity_list
source_user = 'S1'
target_user = 'S2'  # S3

Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5


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
print()
S_label = [int(x) for x in S_label]
T_label = [int(x) for x in T_label]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model training paras settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-2
num_D = 6
width = int(Sampling_frequency * Num_Seconds)
Num_classes = 4
Epochs = 200
Local_epoch = 1
cuda = False
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# DANN model
S_torch_loader = get_DANN_data(S_data, S_label, batch_size=10000, num_D=num_D, width=width)
T_torch_loader = get_DANN_data(T_data, T_label, batch_size=10000, num_D=num_D, width=width)
Kernel_size = 9
Second_dim = int(((width - Kernel_size + 1) / 2 - Kernel_size + 1) / 2)
DANN_train(S_torch_loader, T_torch_loader, cuda, lr, Epochs, num_class=Num_classes, kernel_size=Kernel_size, second_dim=Second_dim, model_root='models')
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

