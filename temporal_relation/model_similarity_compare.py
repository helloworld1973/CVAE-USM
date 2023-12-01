import joblib
import cca_core
from CKA import linear_CKA, kernel_CKA
import numpy as np
from scipy.spatial import distance

S_MRF_dict = joblib.load('S_MRF_dict')
T_MRF_dict = joblib.load('T_MRF_dict')
print()
num_len = len(S_MRF_dict)
layers = 12

for i in range(num_len):
    S_a_act = S_MRF_dict[i]
    S_a_act = list(S_a_act.items())
    for j in range(num_len):
        T_a_act = T_MRF_dict[j]
        T_a_act = list(T_a_act.items())
        for k in [17]: # [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            S_a_act_a_layer = S_a_act[k][1]
            S_a_act_a_layer = S_a_act_a_layer.detach().numpy()
            if len(S_a_act_a_layer.shape) == 1:
                S_a_act_a_layer = S_a_act_a_layer.reshape((S_a_act_a_layer.shape[0], 1))

            T_a_act_a_layer = T_a_act[k][1]
            T_a_act_a_layer = T_a_act_a_layer.detach().numpy()
            if len(T_a_act_a_layer.shape) == 1:
                T_a_act_a_layer = T_a_act_a_layer.reshape((T_a_act_a_layer.shape[0], 1))

            dst = [distance.euclidean(S_a_act_a_layer[:, i], T_a_act_a_layer[:, i]) for i in range(S_a_act_a_layer.shape[1])]
            dst = sum(dst)

            #print('Euclidean, S_' + str(i) + '_VS_T_' + str(j) + '_Layer_' + str(k) + ': {}'.format(dst))

            #print('Linear CKA, S_' + str(i) + '_VS_T_' + str(j) + '_Layer_' + str(k) + ': {}'.format(linear_CKA(S_a_act_a_layer, T_a_act_a_layer)))
            print('RBF Kernel CKA, S_' + str(i) + '_VS_T_' + str(j) + '_Layer_' + str(k) + ': {}'.format( kernel_CKA(S_a_act_a_layer, T_a_act_a_layer)))
            # CCA
            # a_results = cca_core.get_cca_similarity(S_a_act_a_layer, T_a_act_a_layer, epsilon=1e-10, verbose=False)
            # print("Mean CCA similarity" + str(i) + 'T_' + str(j) + 'Layer_' + str(k), np.mean(a_results["cca_coef1"]))
