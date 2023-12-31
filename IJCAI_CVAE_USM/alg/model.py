import torch
from sklearn.cluster import KMeans
from torch import nn, mean
import torch.nn.functional as F
import numpy as np
from IJCAI_CVAE_USM.loss.common_loss import kl_divergence_reserve_structure
from IJCAI_CVAE_USM.network.Adver_network import ReverseLayerF, Discriminator
from IJCAI_CVAE_USM.network.common_network import cvae_encoder, cvae_decoder, cvae_reparameterize, linear_classifier
from IJCAI_CVAE_USM.network.feature_extraction_network import CNN_Feature_Extraction_Network
from scipy.spatial.distance import cdist
from USM import pyusm
import ot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class CVAE_USM(nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num, in_features_size,
                 hidden_size, dis_hidden, num_class, num_temporal_states, reverseLayer_latent_domain_alpha, variance,
                 alpha, beta, gamma, delta, epsilon):
        super(CVAE_USM, self).__init__()

        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_temporal_states = num_temporal_states
        self.ReverseLayer_latent_domain_alpha = reverseLayer_latent_domain_alpha
        self.Variance = variance

        self.Alpha = alpha
        self.Beta = beta
        self.Gamma = gamma
        self.Delta = delta
        self.Epsilon = epsilon

        self.conv1_in_channels = conv1_in_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.kernel_size_num = kernel_size_num
        self.in_features_size = in_features_size
        self.dis_hidden = dis_hidden

        self.featurizer = CNN_Feature_Extraction_Network(self.conv1_in_channels, self.conv1_out_channels,
                                                         self.conv2_out_channels, self.kernel_size_num,
                                                         self.in_features_size)

        self.CVAE_encoder = cvae_encoder(self.in_features_size, self.hidden_size)
        self.CVAE_reparameterize = cvae_reparameterize()
        self.CVAE_decoder = cvae_decoder(self.in_features_size, self.hidden_size)
        self.classify_source = linear_classifier(self.hidden_size, self.num_class)
        self.domains = Discriminator(self.hidden_size, self.dis_hidden, 2)
        # self.temporal_states = linear_classifier(self.hidden_size, self.num_temporal_states)

    def update(self, ST_data, S_data, T_data, opt, device):
        all_x = ST_data[0].float()
        all_c = ST_data[1].long()
        all_ts = ST_data[2].long()
        all_d = ST_data[3].long()

        # VAE update
        all_x_after_fe = self.featurizer(all_x)
        all_mu, all_logvar = self.CVAE_encoder(all_x_after_fe)
        all_z = self.CVAE_reparameterize(all_mu, all_logvar)
        all_x_recon = self.CVAE_decoder(all_z)
        RECON_L = mean((all_x_recon - all_x_after_fe) ** 2)
        KLD_L = kl_divergence_reserve_structure(all_mu, all_logvar, self.Variance)

        # USM update
        predict_all_temporal_state_labels = all_z  # self.temporal_states(all_z)
        # predict_all_temporal_state_labels = F.softmax(predict_all_temporal_state_labels)
        # predict_all_temporal_state_labels = torch.argmax(predict_all_temporal_state_labels, dim=1)
        TEMPORAL_L, temporal_state_labels_S, temporal_state_labels_T = self.USM_temporal_extraction(
            predict_all_temporal_state_labels, ST_data, S_data, T_data, device)

        # domains (users) update
        disc_d_in1 = ReverseLayerF.apply(all_z, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.domains(disc_d_in1)
        disc_DOMAIN_L = F.cross_entropy(disc_d_out1, all_d)

        # source users activity classification update
        S_x = S_data[0].float()
        S_c = S_data[1].long()
        S_ts = S_data[2].long()
        S_d = ST_data[3].long()
        S_x_after_fe = self.featurizer(S_x)
        S_mu, S_logvar = self.CVAE_encoder(S_x_after_fe)
        S_z = self.CVAE_reparameterize(S_mu, S_logvar)
        predict_S_class_labels = self.classify_source(S_z)
        S_CLASS_L = F.cross_entropy(predict_S_class_labels, S_c)

        loss = self.Alpha * RECON_L + self.Beta * KLD_L + self.Gamma * S_CLASS_L + self.Delta * disc_DOMAIN_L + self.Epsilon * TEMPORAL_L
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'reconstruct': RECON_L.item(), 'KL': KLD_L.item(),
                'source_classes': S_CLASS_L.item(), 'disc_domains': disc_DOMAIN_L.item(), 'temporal': TEMPORAL_L.item()}, temporal_state_labels_S, temporal_state_labels_T

    def USM_temporal_extraction(self, predict_all_temporal_state_labels, ST_data, S_data, T_data, device):
        # generate USM features for each activity and each user
        # predict_all_temporal_state_labels_list = predict_all_temporal_state_labels.tolist()
        predict_all_temporal_state_numpy_data = predict_all_temporal_state_labels.detach().numpy()
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=self.num_temporal_states, random_state=0)
        kmeans.fit(predict_all_temporal_state_numpy_data)
        # Get the cluster labels and convert to a PyTorch tensor
        cluster_labels = kmeans.labels_

        predict_all_temporal_state_labels_list = cluster_labels.tolist()

        S_labels = S_data[1].tolist()
        T_labels = T_data[1].tolist()
        unique_labels = sorted(set(S_labels))
        s_len = len(S_labels)

        s_sequences = []
        t_sequences = []
        for i in unique_labels:
            S_a_activity_index = [index for index, element in enumerate(S_labels) if element == i]
            S_a_activity_temporal_states = [predict_all_temporal_state_labels_list[index] for index in
                                            S_a_activity_index]
            s_sequences.append(S_a_activity_temporal_states)

            T_a_activity_index = [index for index, element in enumerate(T_labels) if element == i]
            T_a_activity_temporal_states = [predict_all_temporal_state_labels_list[index + s_len] for index in
                                            T_a_activity_index]
            t_sequences.append(T_a_activity_temporal_states)

        # Extract coordinates from the USM object
        s_USM = []
        t_USM = []
        usm_data_s = pyusm.USM.make_usm(predict_all_temporal_state_labels_list)
        usm_baseline_coordinates = usm_data_s.coord_dict
        for a_sequence in s_sequences:
            usm_data = pyusm.USM.make_usm(a_sequence, A=usm_baseline_coordinates)
            s_USM.append(usm_data.fw)
        for a_sequence in t_sequences:
            usm_data = pyusm.USM.make_usm(a_sequence, A=usm_baseline_coordinates)
            t_USM.append(usm_data.fw)

        '''
        # Using t-SNE to reduce dimensionality to 2D for plotting
        tsne = TSNE(n_components=2, random_state=0)

        # Concatenating all arrays for t-SNE transformation
        all_data = np.concatenate(s_USM)
        all_data_2d = tsne.fit_transform(all_data)

        # Extracting the transformed points for each array
        split_indices = np.cumsum([len(arr) for arr in s_USM])[:-1]
        s_USM_2d = np.split(all_data_2d, split_indices)

        # Plotting
        colors = plt.cm.get_cmap('tab20', 11)
        plt.figure(figsize=(10, 6))
        for i, data in enumerate(s_USM_2d):
            plt.scatter(data[:, 0], data[:, 1], color=colors(i), label=f'Set {i + 1}')

        plt.title('t-SNE Visualization of s_USM')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.show()
        '''

        # apply Wasserstein distance to get temporal relation loss
        s_USM = np.concatenate(s_USM)
        kmeans_after_USM_S = KMeans(n_clusters=self.num_temporal_states, random_state=0)
        kmeans_after_USM_S.fit(s_USM)

        t_USM = np.concatenate(t_USM)
        kmeans_after_USM_T = KMeans(n_clusters=self.num_temporal_states, random_state=0)
        kmeans_after_USM_T.fit(t_USM)

        # Get the cluster centers
        centers_S = kmeans_after_USM_S.cluster_centers_
        centers_T = kmeans_after_USM_T.cluster_centers_
        M = ot.dist(centers_S, centers_T, metric='euclidean')
        a = ot.unif(centers_S.shape[0])
        b = ot.unif(centers_T.shape[0])
        wasserstein_distance = ot.emd2(a, b, M)
        wasserstein_dist_tensor = torch.tensor(wasserstein_distance, dtype=torch.float32)

        # update
        temporal_state_labels_S = kmeans_after_USM_S.labels_
        temporal_state_labels_T = kmeans_after_USM_T.labels_

        return wasserstein_dist_tensor, temporal_state_labels_S, temporal_state_labels_T

    def predict(self, x):
        mu, _ = self.CVAE_encoder(self.featurizer(x))
        return self.classify_source(mu), mu

    def GPU_set_tlabel(self, S_torch_loader, T_torch_loader, temporal_state_labels_S, temporal_state_labels_T, device):
        self.CVAE_encoder.eval()
        self.CVAE_reparameterize.eval()
        self.featurizer.eval()

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0].to(device), S_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(temporal_state_labels_S).to(device),
            S_torch_loader.dataset.tensors[3].to(device), S_torch_loader.dataset.tensors[4].to(device))

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0].to(device), T_torch_loader.dataset.tensors[1].to(device),
            torch.tensor(temporal_state_labels_T).to(device),
            T_torch_loader.dataset.tensors[3].to(device), T_torch_loader.dataset.tensors[4].to(device))

        self.CVAE_encoder.train()
        self.CVAE_reparameterize.train()
        self.featurizer.train()
