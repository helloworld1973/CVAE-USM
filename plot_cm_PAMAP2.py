import matplotlib.pyplot as plt
import numpy as np

# 1-5
one_confusion_matrix_data = """
0.0	149.0	7.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
38.0	132.0	7.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	146.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	7.0	198.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
4.0	0.0	0.0	0.0	22.0	0.0	136.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	155.0	0.0	0.0	0.0	0.0	7.0
0.0	0.0	7.0	1.0	0.0	0.0	165.0	0.0	0.0	0.0	0.0
0.0	0.0	7.0	0.0	0.0	0.0	0.0	79.0	0.0	8.0	0.0
0.0	0.0	9.0	0.0	0.0	0.0	0.0	0.0	0.0	74.0	0.0
0.0	0.0	2.0	0.0	0.0	0.0	2.0	0.0	0.0	157.0	0.0
0.0	0.0	7.0	0.0	0.0	4.0	0.0	0.0	0.0	0.0	207.0
"""
confusion_matrix_1 = np.array(
    [list(map(float, row.split())) for row in one_confusion_matrix_data.strip().split('\n')])

# 1-6
two_confusion_matrix_data = """
154.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	152.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	141.0	20.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	163.0	0.0	0.0	0.0	0.0	0.0	6.0	0.0
0.0	0.0	0.0	0.0	148.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	135.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	162.0	14.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	87.0	0.0	0.0	0.0
3.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	71.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	139.0	0.0
0.0	158.0	0.0	0.0	0.0	92.0	0.0	0.0	0.0	0.0	0.0
"""
confusion_matrix_2 = np.array(
    [list(map(float, row.split())) for row in two_confusion_matrix_data.strip().split('\n')])

# 5-1
three_confusion_matrix_data = """
1000.0	80.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	155.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	143.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	139.0	0.0	0.0	0.0	0.0	0.0	0.0	4.0
0.0	0.0	0.0	0.0	123.0	0.0	10.0	0.0	0.0	0.0	6.0
0.0	0.0	0.0	0.0	0.0	137.0	0.0	0.0	0.0	0.0	18.0
0.0	0.0	0.0	0.0	0.0	0.0	121.0	0.0	0.0	0.0	12.0
0.0	0.0	0.0	63.0	0.0	0.0	0.0	33.0	0.0	0.0	8.0
1.0	0.0	0.0	91.0	0.0	0.0	0.0	0.0	0.0	0.0	6.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	144.0	7.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	156.0
"""
confusion_matrix_3 = np.array(
    [list(map(float, row.split())) for row in three_confusion_matrix_data.strip().split('\n')])


# 5-6
four_confusion_matrix_data = """
149.0	0.0	5.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
9.0	140.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	3.0	0.0
0.0	0.0	161.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	3.0	165.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
0.0	0.0	4.0	6.0	136.0	0.0	2.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	66.0	0.0	0.0	0.0	69.0	0.0
0.0	4.0	0.0	0.0	0.0	0.0	171.0	0.0	0.0	1.0	0.0
0.0	0.0	13.0	74.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	69.0	5.0
0.0	0.0	0.0	23.0	0.0	0.0	0.0	0.0	0.0	116.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	3.0	247.0	
"""
confusion_matrix_4 = np.array(
    [list(map(float, row.split())) for row in four_confusion_matrix_data.strip().split('\n')])

# 6-1
five_confusion_matrix_data = """
180.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	126.0	11.0	0.0	0.0	0.0	0.0	0.0	0.0	9.0	9.0
0.0	6.0	137.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	110.0	0.0	0.0	0.0	19.0	0.0	0.0	14.0
0.0	0.0	0.0	0.0	139.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	155.0
0.0	0.0	0.0	0.0	0.0	0.0	79.0	54.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	10.0	90.0	0.0	0.0	4.0
0.0	0.0	1.0	0.0	0.0	74.0	0.0	0.0	0.0	19.0	4.0
0.0	0.0	0.0	0.0	0.0	17.0	0.0	0.0	0.0	134.0	0.0
0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	26.0	0.0	129.0
"""
confusion_matrix_5 = np.array(
    [list(map(float, row.split())) for row in five_confusion_matrix_data.strip().split('\n')])

# 6-5
six_confusion_matrix_data = """
0.0	141.0	8.0	0.0	0.0	0.0	0.0	0.0	0.0	6.0	1.0
27.0	144.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	6.0	0.0
0.0	0.0	102.0	0.0	0.0	0.0	0.0	0.0	0.0	44.0	0.0
0.0	0.0	0.0	198.0	0.0	0.0	0.0	0.0	0.0	7.0	0.0
0.0	0.0	0.0	6.0	156.0	0.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	7.0	0.0	0.0	155.0	0.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	173.0	0.0	0.0	0.0	0.0
0.0	0.0	0.0	0.0	0.0	0.0	0.0	88.0	0.0	6.0	0.0
13.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	69.0	0.0	0.0
4.0	0.0	0.0	48.0	0.0	74.0	0.0	0.0	35.0	0.0	0.0
0.0	7.0	0.0	0.0	0.0	0.0	0.0	0.0	12.0	17.0	182.0
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

activity_list = ['lying', 'sitting', 'standing', 'walking', 'running',
                 'cycling', 'Nordic walking', 'ascending stairs', 'descending stairs',
                 'vacuum cleaning', 'ironing']

activity_list = ['1', '2', '3', '4', '5',
                 '6', '7', '8', '9',
                 '10', '11']

ax.set_xticks(np.arange(len(activity_list)))
ax.set_yticks(np.arange(len(activity_list)))
ax.set_xticklabels(activity_list, rotation=90, fontsize=28)
ax.set_yticklabels(activity_list, fontsize=28)

# Saving the figure to a file
file_path = "pamap2_cm_CVAE_USM_results.png"
plt.tight_layout()
plt.savefig(file_path)
plt.show()
plt.close()
