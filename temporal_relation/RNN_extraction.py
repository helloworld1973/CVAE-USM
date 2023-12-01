import joblib
import numpy as np
import torch
from torch import nn
from gtda.time_series import SlidingWindow
from read_dataset import read_OPPT_dataset
from sklearn import preprocessing

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm
activity_list = ['Stand', 'Walk', 'Sit', 'Lie']
activities_required = activity_list
# Sampling_frequency = 30  # HZ
N_input = 6
DATASET_NAME = 'OPPT'


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def data_read(source_user='S1', target_user='S3', N_steps=128, Window_Overlap_Rate=0.5):
    # OPPT_dataset
    sensor_channels_required = ['IMU_BACK_ACC_X', 'IMU_BACK_ACC_Y', 'IMU_BACK_ACC_Z',
                                'IMU_BACK_GYRO_X', 'IMU_BACK_GYRO_Y', 'IMU_BACK_GYRO_Z']  # right lower arm
    activity_list = ['Stand', 'Walk', 'Sit', 'Lie']
    activities_required = activity_list
    # Sampling_frequency = 30  # HZ
    DATASET_NAME = 'OPPT'
    oppt_ds = read_OPPT_dataset.READ_OPPT_DATASET(source_user, target_user, n_steps=N_steps,
                                                  bag_overlap_rate=Window_Overlap_Rate)

    source_required_X_bags, source_required_Y_bags, source_required_amount, \
    target_required_X_bags, target_required_Y_bags, target_required_amount, \
    source_data_x, target_data_x, source_data_y, target_data_y \
        = oppt_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required,
                                                                             activities_required)
    source_data_y = [i - 1 for i in source_data_y]
    target_data_y = [i - 1 for i in target_data_y]

    return source_data_x, source_data_y, target_data_x, target_data_y


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model design
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size

        self.rnn_0 = nn.GRU(input_size=self.input_size, hidden_size=self.input_size * 2, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_0 = torch.nn.LayerNorm(self.input_size * 2, eps=0)
        self.ln_0.training = False
        self.ln_0 = self.ln_0.double()

        self.rnn_1 = nn.GRU(input_size=self.input_size * 2, hidden_size=self.input_size * 4, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_1 = torch.nn.LayerNorm(self.input_size * 4, eps=0)
        self.ln_1.training = False
        self.ln_1 = self.ln_1.double()

        self.rnn_2 = nn.GRU(input_size=self.input_size * 4, hidden_size=self.input_size * 8, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_2 = torch.nn.LayerNorm(self.input_size * 8, eps=0)
        self.ln_2.training = False
        self.ln_2 = self.ln_2.double()

        self.rnn_3 = nn.GRU(input_size=self.input_size * 8, hidden_size=self.input_size * 4, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_3 = torch.nn.LayerNorm(self.input_size * 4, eps=0)
        self.ln_3.training = False
        self.ln_3 = self.ln_3.double()

        self.rnn_4 = nn.GRU(input_size=self.input_size * 4, hidden_size=self.input_size * 2, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_4 = torch.nn.LayerNorm(self.input_size * 2, eps=0)
        self.ln_4.training = False
        self.ln_4 = self.ln_4.double()

        self.rnn_5 = nn.GRU(input_size=self.input_size * 2, hidden_size=self.input_size, num_layers=1,
                            batch_first=True, dropout=0, bias=False)
        self.ln_5 = torch.nn.LayerNorm(self.input_size, eps=0)
        self.ln_5.training = False
        self.ln_5 = self.ln_5.double()

    def forward(self, input):
        output_0, hidden_0 = self.rnn_0(input)
        output_ln_0 = self.ln_0(output_0)

        output_1, hidden_1 = self.rnn_1(output_ln_0)
        output_ln_1 = self.ln_1(output_1)

        output_2, hidden_2 = self.rnn_2(output_ln_1)
        output_ln_2 = self.ln_2(output_2)

        output_3, hidden_3 = self.rnn_3(output_ln_2)
        output_ln_3 = self.ln_3(output_3)

        output_4, hidden_4 = self.rnn_4(output_ln_3)
        output_ln_4 = self.ln_4(output_4)

        output_5, hidden_5 = self.rnn_5(output_ln_4)
        output_ln_5 = self.ln_5(output_5)

        return output_ln_5

    def init_weight(self):
        # orthogonal initialization of recurrent weights
        for name, param in self.named_parameters():
            if 'weight' in name and 'gru' in name:
                nn.init.orthogonal_(param, gain=1)


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


N_Classes = len(activity_list)
N_Layer = 2
Hidden_State = 6
n_steps = 30
bag_overlap_rate = 0.9
source_data_x, source_data_y, target_data_x, target_data_y = data_read(source_user='S1', target_user='S3', N_steps=30,
                                                                       Window_Overlap_Rate=0.9)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
S_labels = source_data_y
S_data = source_data_x
S_data = np.array(S_data)
scaler = preprocessing.StandardScaler().fit(S_data)
S_data = scaler.transform(S_data)
S_MRF_dict = []
for index in range(len(activity_list)):
    get_a_index = [x for x, y in enumerate(S_labels) if y == index]
    a_act_data_list = [S_data[j, :] for j in get_a_index]
    a_act_data_list = np.array(a_act_data_list)
    a_act_data_y_list = [S_labels[j] for j in get_a_index]
    sliding_bag = SlidingWindow(size=int(n_steps), stride=int(n_steps * (1 - bag_overlap_rate)))
    X_bags = sliding_bag.fit_transform(a_act_data_list)
    Y_bags = sliding_bag.resample(a_act_data_y_list)  # last occur label
    Y_bags = Y_bags.tolist()
    print()
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # train model
    input_sequence = X_bags[0:-1, :, :]
    baseline_sequence = X_bags[1:, :, :]
    input_sequence = torch.from_numpy(np.array(input_sequence).astype(float))
    baseline_sequence = torch.from_numpy(np.array(baseline_sequence).astype(float))
    # Instantiate the model with hyperparameters
    model = Model(input_size=N_input)
    model = model.double()
    model.init_weight()
    # Define hyperparameters
    N_Epochs = 2000
    Learn_Rate = 0.01
    # Define Loss, Optimizer
    criterion_temporal = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learn_Rate)
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print()
    for epoch in range(1, N_Epochs + 1):
        optimizer.zero_grad()
        output = model(input_sequence)
        loss_temporal = criterion_temporal(output, baseline_sequence)
        loss_temporal.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, N_Epochs), end=' ')
            print("source_temporal_loss: {:.4f}".format(loss_temporal.item()))
    S_MRF_dict.append(model.state_dict())
    print()
joblib.dump(S_MRF_dict, 'S_MRF_dict')

# -------------------------------

T_labels = target_data_y
T_data = target_data_x
T_data = np.array(T_data)
scaler = preprocessing.StandardScaler().fit(T_data)
T_data = scaler.transform(T_data)
T_MRF_dict = []
for index in range(len(activity_list)):
    get_a_index = [x for x, y in enumerate(T_labels) if y == index]
    a_act_data_list = [T_data[j, :] for j in get_a_index]
    a_act_data_list = np.array(a_act_data_list)
    a_act_data_y_list = [T_labels[j] for j in get_a_index]
    sliding_bag = SlidingWindow(size=int(n_steps), stride=int(n_steps * (1 - bag_overlap_rate)))
    X_bags = sliding_bag.fit_transform(a_act_data_list)
    Y_bags = sliding_bag.resample(a_act_data_y_list)  # last occur label
    Y_bags = Y_bags.tolist()
    print()
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # train model
    input_sequence = X_bags[0:-1, :, :]
    baseline_sequence = X_bags[1:, :, :]
    input_sequence = torch.from_numpy(np.array(input_sequence).astype(float))
    baseline_sequence = torch.from_numpy(np.array(baseline_sequence).astype(float))
    # Instantiate the model with hyperparameters
    model = Model(input_size=N_input)
    model = model.double()
    model.init_weight()
    # Define hyperparameters
    N_Epochs = 2000
    Learn_Rate = 0.01
    # Define Loss, Optimizer
    criterion_temporal = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learn_Rate)
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print()
    for epoch in range(1, N_Epochs + 1):
        optimizer.zero_grad()
        output = model(input_sequence)
        loss_temporal = criterion_temporal(output, baseline_sequence)
        loss_temporal.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, N_Epochs), end=' ')
            print("target_temporal_loss: {:.4f}".format(loss_temporal.item()))
    T_MRF_dict.append(model.state_dict())
    print()
joblib.dump(T_MRF_dict, 'T_MRF_dict')
print()


