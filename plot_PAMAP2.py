import matplotlib.pyplot as plt
import numpy as np

methods = ['CORAL', 'SOT', 'DANN', 'TrC', 'CVAE-USM']
values_1_5 = [32.69, 69.82, 61.66, 62.58, 72.6]
values_1_6 = [42.62, 57.8, 70.33, 70.33, 70.52]
values_5_1 = [50.95, 64.54, 60.03, 64.10, 73.92]
values_5_6 = [57.14, 61.95, 68.36, 76.05, 82.12]
values_6_1 = [59.04, 62.48, 68.17, 64.48, 72.19]
values_6_5 = [38.25, 56.14, 61.35, 64.94, 72.94]

# Compute the average accuracy for each method
averages = [
    np.mean([values_1_5[i], values_1_6[i], values_5_1[i], values_5_6[i], values_6_1[i], values_6_5[i]])
    for i in range(len(methods))
]

barWidth = 0.12
r1 = np.arange(len(values_1_5))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]  # New set of positions for average bars

plt.bar(r1, values_1_5, width=barWidth, label='1 → 5')
plt.bar(r2, values_1_6, width=barWidth, label='1 → 6')
plt.bar(r3, values_5_1, width=barWidth, label='5 → 1')
plt.bar(r4, values_5_6, width=barWidth, label='5 → 6')
plt.bar(r5, values_6_1, width=barWidth, label='6 → 1')
plt.bar(r6, values_6_5, width=barWidth, label='6 → 5')
plt.bar(r7, averages, width=barWidth, color='gray', alpha=0.7, label='Average')  # Bar for average values

plt.xlabel('Methods', fontweight='bold')
plt.xticks([r + 3*barWidth for r in range(len(values_1_5))], methods)  # Adjust x-tick positions
plt.ylabel('Accuracy (%)')
plt.title('PAMAP2 Dataset Results')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()

plt.savefig('pamap2_results_with_avg.png', dpi=300)
plt.show()
