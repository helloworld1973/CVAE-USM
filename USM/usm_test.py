import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pyusm
import matplotlib.pyplot as plt

data1 = list('ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABC')
data2 = list('CBACBACBACBACBACBACBACBACBACBACBACBACBACBA')


usm_data1 = pyusm.USM.make_usm(data1)
print()

# Extract coordinates from the USM object
coordinates_usm_data1 = usm_data1.fw

usm_data2 = pyusm.USM.make_usm(data2, A=usm_data1.coord_dict)

coordinates_usm_data2 = usm_data2.fw

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=5, n_iter=300)  # Adjust perplexity and n_iter as needed
coordinates_combined = np.concatenate((coordinates_usm_data1, coordinates_usm_data2), axis=0)
reduced_coordinates_combined = tsne.fit_transform(coordinates_combined)

# Create a figure
plt.figure(figsize=(10, 8))

# Plot the reduced data points for data1 (in blue) and label with 'ABCDEF'
for i in range(len(data1)):
    x, y = reduced_coordinates_combined[i]
    label = data1[i]
    plt.scatter(x, y, s=50, c='blue')
    plt.text(x, y, label, ha='center', va='bottom')

# Plot the reduced data points for data2 (in red) and label with 'ADCEF'
for i in range(len(data2)):
    x, y = reduced_coordinates_combined[i + len(data1)]
    label = data2[i]
    plt.scatter(x, y, s=50, c='red')
    plt.text(x, y, label, ha='center', va='bottom')

# Add labels, titles, and legend
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Combined Datasets with USM')
plt.legend()

# Show the figure
plt.show()
print()