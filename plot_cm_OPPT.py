import matplotlib.pyplot as plt
import numpy as np

# S1-S2
one_confusion_matrix_data = """
468.0	0.0	0.0	0.0
0.0	660.0	0.0	0.0
0.0	0.0	398.0	0.0
0.0	1.0	0.0	498.0
"""
confusion_matrix_1 = np.array(
    [list(map(float, row.split())) for row in one_confusion_matrix_data.strip().split('\n')])

# S1-S3
two_confusion_matrix_data = """
226.0	0.0	0.0	0.0
0.0	742.0	0.0	0.0
0.0	0.0	467.0	0.0
0.0	0.0	0.0	271.0
"""
confusion_matrix_2 = np.array(
    [list(map(float, row.split())) for row in two_confusion_matrix_data.strip().split('\n')])

# S2-S1
three_confusion_matrix_data = """
486.0	0.0	0.0	0.0
0.0	686.0	0.0	0.0
0.0	0.0	422.0	0.0
0.0	0.0	0.0	448.0
"""
confusion_matrix_3 = np.array(
    [list(map(float, row.split())) for row in three_confusion_matrix_data.strip().split('\n')])


# S2-S3
four_confusion_matrix_data = """
226.0	0.0	0.0	0.0
6.0	736.0	0.0	0.0
6.0	0.0	461.0	0.0
0.0	0.0	5.0	266.0
"""
confusion_matrix_4 = np.array(
    [list(map(float, row.split())) for row in four_confusion_matrix_data.strip().split('\n')])

# S3-S1
five_confusion_matrix_data = """
486.0	0.0	0.0	0.0
0.0	686.0	0.0	0.0
0.0	0.0	422.0	0.0
0.0	0.0	0.0	448.0
"""
confusion_matrix_5 = np.array(
    [list(map(float, row.split())) for row in five_confusion_matrix_data.strip().split('\n')])

# S3-S2
six_confusion_matrix_data = """
432.0	0.0	36.0	0.0
0.0	610.0	50.0	0.0
0.0	12.0	386.0	0.0
56.0	0.0	0.0	443.0
"""
confusion_matrix_6 = np.array(
    [list(map(float, row.split())) for row in six_confusion_matrix_data.strip().split('\n')])

def compute_accuracy_from_confusion_matrix(confusion_matrix):
    # Normalize the confusion matrix
    normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100

    # Compute the accuracy
    accuracy = np.trace(normalized_confusion_matrix) / (100 * confusion_matrix.shape[0])

    return accuracy


confusion_matrix = (np.array(confusion_matrix_1) + np.array(confusion_matrix_2) + np.array(confusion_matrix_3) +
                    np.array(confusion_matrix_4) + np.array(confusion_matrix_5) + np.array(confusion_matrix_6)) / 6

confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100
# Removing the colorbar and saving the figure to a file

acc = compute_accuracy_from_confusion_matrix(confusion_matrix)


fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(confusion_matrix, cmap='viridis')
# plt.colorbar(cax)

# Setting titles and labels
#plt.title('Confusion Matrix', pad=20)
plt.xlabel('Predicted Activity', fontsize=28)
plt.ylabel('True Activity', fontsize=28)

activity_list = ['standing', 'walking', 'sitting', 'lying']

activity_list = ['1', '2', '3', '4']

ax.set_xticks(np.arange(len(activity_list)))
ax.set_yticks(np.arange(len(activity_list)))
ax.set_xticklabels(activity_list, rotation=90, fontsize=28)
ax.set_yticklabels(activity_list, fontsize=28)

# Saving the figure to a file
file_path = "OPPT_cm_CVAE_USM_results.png"
plt.tight_layout()
plt.savefig(file_path)
plt.show()
plt.close()
