import torch.nn as nn
from DANN.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self, conv1_in_channels=6, conv1_out_channels=16, conv2_out_channels=32, full_connect_num=100,
                 num_class=4, kernel_size=9, second_dim=69):
        super(CNNModel, self).__init__()
        self.conv1_in_channels = conv1_in_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.kernel_size = kernel_size
        self.full_connect_num = full_connect_num
        self.num_class = num_class
        self.second_dim =second_dim

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(self.conv1_in_channels, self.conv1_out_channels,
                                                     kernel_size=(1, self.kernel_size)))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(self.conv1_out_channels))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=(1, 2), stride=2))

        self.feature.add_module('f_conv2', nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels,
                                                     kernel_size=(1, self.kernel_size)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(self.conv2_out_channels))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=(1, 2), stride=2))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.conv2_out_channels * self.second_dim, self.full_connect_num))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.full_connect_num))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(self.full_connect_num, self.full_connect_num))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(self.full_connect_num))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(self.full_connect_num, self.num_class))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.conv2_out_channels * self.second_dim, self.full_connect_num))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.full_connect_num))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.full_connect_num, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, self.conv2_out_channels * feature.size(-1))
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
