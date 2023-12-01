import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow


# sampling frequency: 100Hz
class READ_PAMAP2_DATASET:
    def __init__(self, source_user, target_user, n_steps, bag_overlap_rate):
        self.File_Path = 'E:\\Python Projects\\TL_DATASETS_REPO\\PAMAP2_Dataset\\Protocol\\'
        self.source_user = source_user
        self.target_user = target_user
        self.n_steps = n_steps
        self.bag_overlap_rate = bag_overlap_rate
        self.Activity_Mapping = {'transient': 0,
                                 'lying': 1,
                                 'sitting': 2,
                                 'standing': 3,
                                 'walking': 4,
                                 'running': 5,
                                 'cycling': 6,
                                 'Nordic_walking': 7,
                                 'watching_TV': 9,
                                 'computer_work': 10,
                                 'car driving': 11,
                                 'ascending_stairs': 12,
                                 'descending_stairs': 13,
                                 'vacuum_cleaning': 16,
                                 'ironing': 17,
                                 'folding_laundry': 18,
                                 'house_cleaning': 19,
                                 'playing_soccer': 20,
                                 'rope_jumping': 24}
        '''
        self.Activity_Mapping = {'lying': 0,
                                 'sitting': 1,
                                 'standing': 2,
                                 'walking': 3,
                                 'running': 4,
                                 'cycling': 5,
                                 'Nordic_walking': 6,
                                 'ascending_stairs': 7,
                                 'descending_stairs': 8,
                                 'vacuum_cleaning': 9,
                                 'ironing': 10,
                                 'rope_jumping': 11}
        '''
        self.IMU_Hand = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3',
                         'handGyro1', 'handGyro2', 'handGyro3',
                         'handMagne1', 'handMagne2', 'handMagne3']
        self.IMU_Chest = ['chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                          'chestGyro1', 'chestGyro2', 'chestGyro3',
                          'chestMagne1', 'chestMagne2', 'chestMagne3']
        self.IMU_Ankle = ['ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                          'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                          'ankleMagne1', 'ankleMagne2', 'ankleMagne3']
        self.Col_Names = ["timestamp", "activityID"]

        self.Sensor_Dict = {'IMU_Hand': self.IMU_Hand,
                            'IMU_Chest': self.IMU_Chest,
                            'IMU_Ankle': self.IMU_Ankle}
        self.complete_columns = ["timestamp", "activityID", "heartrate"] + ['handTemperature',
                                                                            'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
                                                                            'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                                                                            'handGyro1', 'handGyro2', 'handGyro3',
                                                                            'handMagne1', 'handMagne2', 'handMagne3',
                                                                            'handOrientation1', 'handOrientation2',
                                                                            'handOrientation3', 'handOrientation4'] + [
                                    'chestTemperature',
                                    'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                                    'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                                    'chestGyro1', 'chestGyro2', 'chestGyro3',
                                    'chestMagne1', 'chestMagne2', 'chestMagne3',
                                    'chestOrientation1', 'chestOrientation2', 'chestOrientation3',
                                    'chestOrientation4'] + ['ankleTemperature',
                                                            'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                                                            'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                                                            'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                                                            'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                                                            'ankleOrientation1', 'ankleOrientation2',
                                                            'ankleOrientation3', 'ankleOrientation4']

    def find_activity_ID_by_activity_name(self, activity_name):
        activity_ID = self.Activity_Mapping[activity_name]
        return activity_ID

    def find_sensor_channel(self, sensor_channel_name):
        sensor_channels = self.Sensor_Dict[sensor_channel_name]
        return sensor_channels

    def generate_data_with_required_sensor_channels_and_activities_and_a_user(self, which_user, columns,
                                                                              activities_ID_list):
        file_path = self.File_Path + 'subject10' + which_user + '.dat'
        df = pd.DataFrame()
        data_tale = pd.read_table(file_path, header=None, sep=r'\s+')
        data_tale.columns = self.complete_columns
        df = df.append(data_tale, ignore_index=True)
        df = df.loc[:, columns]

        # df.reset_index(drop=True, inplace=True)
        # remove null Nan value
        print(df.isnull().sum())
        is_NaN = df.isnull()
        row_has_NaN = is_NaN.any(axis=1)
        rows_with_NaN = df[row_has_NaN]
        print(rows_with_NaN)
        df = df.dropna()
        print(df.isnull().sum())

        data_x = df.loc[:, df.columns != 'activityID']
        data_x = np.array(data_x.loc[:, data_x.columns != 'timestamp'].values.tolist())
        data_y = df.loc[:, 'activityID']
        data_y = data_y.values.tolist()

        required_X_bags, required_Y_bags, required_amount, data_x, data_y = self.data_segment(
            data_x, data_y, activities_ID_list)

        return required_X_bags, required_Y_bags, required_amount, data_x, data_y

    def data_segment(self, data_x, data_y, activities_ID_list):
        '''
        unify unit as
        gyr: rad / s
        acc: m / s^2
        mag: mT
        '''
        # data_x[:, 6] = data_x[:, 6] / 1000
        # data_x[:, 7] = data_x[:, 7] / 1000
        # data_x[:, 8] = data_x[:, 8] / 1000

        sliding_bag = SlidingWindow(size=int(self.n_steps), stride=int(self.n_steps * (1 - self.bag_overlap_rate)))
        X_bags = sliding_bag.fit_transform(data_x)
        Y_bags = sliding_bag.resample(data_y)  # last occur label
        Y_bags = Y_bags.tolist()

        required_X_bags = []
        required_Y_bags = []
        required_amount = []
        for a_activity_ID in activities_ID_list:
            a_activity_index_list = [i for i, j in enumerate(Y_bags) if j == a_activity_ID]
            X_bags_a_activity = X_bags[a_activity_index_list]
            Y_bags_a_activity = np.array(Y_bags)[a_activity_index_list]

            if Y_bags_a_activity.tolist()[0] in [1, 2, 3, 4, 5, 6, 7]:
                Y_bags_a_activity = Y_bags_a_activity - 1
            elif Y_bags_a_activity.tolist()[0] in [12, 13]:
                Y_bags_a_activity = Y_bags_a_activity - 5
            elif Y_bags_a_activity.tolist()[0] in [16, 17]:
                Y_bags_a_activity = Y_bags_a_activity - 7
            '''
            # all 11 activities
            if Y_bags_a_activity.tolist()[0] in [1, 2, 3, 4, 5, 6, 7]:
                Y_bags_a_activity = Y_bags_a_activity - 1
            elif Y_bags_a_activity.tolist()[0] in [12, 13]:
                Y_bags_a_activity = Y_bags_a_activity - 5
            elif Y_bags_a_activity.tolist()[0] in [16, 17]:
                Y_bags_a_activity = Y_bags_a_activity - 7
            elif Y_bags_a_activity.tolist()[0] in [24]:
                Y_bags_a_activity = Y_bags_a_activity - 13
            '''

            required_X_bags.extend(X_bags_a_activity)
            required_Y_bags.extend(Y_bags_a_activity)
            required_amount.append(len(Y_bags_a_activity))

        raw_x = []
        raw_y = []
        data_y = np.array(data_y)
        for a_activity_ID in activities_ID_list:
            a_activity_index_list = [i for i, j in enumerate(data_y) if j == a_activity_ID]
            a_activity_raw_x = data_x[a_activity_index_list]
            a_activity_raw_y = data_y[a_activity_index_list]
            raw_x.extend(a_activity_raw_x)
            raw_y.extend(a_activity_raw_y)

        raw_y = [0 if x == 1 else x for x in raw_y]
        raw_y = [1 if x == 2 else x for x in raw_y]
        raw_y = [2 if x == 3 else x for x in raw_y]
        raw_y = [3 if x == 4 else x for x in raw_y]
        raw_y = [4 if x == 5 else x for x in raw_y]
        raw_y = [5 if x == 6 else x for x in raw_y]
        raw_y = [6 if x == 7 else x for x in raw_y]
        raw_y = [7 if x == 12 else x for x in raw_y]
        raw_y = [8 if x == 13 else x for x in raw_y]
        raw_y = [9 if x == 16 else x for x in raw_y]
        raw_y = [10 if x == 17 else x for x in raw_y]

        return required_X_bags, required_Y_bags, required_amount, raw_x, raw_y

    def generate_data_with_required_sensor_channels_and_activities(self, sensor_channels_required, activities_required):

        columns = self.Col_Names
        for a_position_channels in sensor_channels_required:
            columns += self.find_sensor_channel(a_position_channels)

        columns = [columns[i] for i in [0,1,2,3,4,5,6,7,11,12,13,14,15,16,20,21,22,23,24,25]]  # only need 6 channels of acc, gyro. remove magnetic

        activities_ID_list = [self.find_activity_ID_by_activity_name(a_activity_name) for a_activity_name in
                              activities_required]

        source_required_X_bags, source_required_Y_bags, source_required_amount, source_data_x, source_data_y = self.generate_data_with_required_sensor_channels_and_activities_and_a_user(
            self.source_user,
            columns,
            activities_ID_list)
        target_required_X_bags, target_required_Y_bags, target_required_amount, target_data_x, target_data_y = self.generate_data_with_required_sensor_channels_and_activities_and_a_user(
            self.target_user,
            columns,
            activities_ID_list)

        return source_required_X_bags, source_required_Y_bags, source_required_amount, \
               target_required_X_bags, target_required_Y_bags, target_required_amount, \
               source_data_x, target_data_x, source_data_y, target_data_y
