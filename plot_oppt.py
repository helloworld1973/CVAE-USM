import matplotlib.pyplot as plt
import numpy as np

methods = ['CORAL', 'SOT', 'DANN', 'TrC', 'CVAE-USM']
values_s1_s2 = [74.18, 75.6, 76.42, 83.04, 99.95]
values_s1_s3 = [84.74, 79.94, 74.07, 83.25, 100.00]
values_s2_s1 = [80.36, 82.76, 82.59, 81.5, 100.00]
values_s2_s3 = [84.74, 73.62, 79.61, 85.43, 99.00]
values_s3_s1 = [78.67, 74.09, 83.03, 82.33, 100.00]
values_s3_s2 = [75.92, 69.53, 79.23, 81.11, 92.4]

# Compute the average accuracy for each method
averages = [
    np.mean([values_s1_s2[i], values_s1_s3[i], values_s2_s1[i], values_s2_s3[i], values_s3_s1[i], values_s3_s2[i]])
    for i in range(len(methods))
]

barWidth = 0.12
r1 = np.arange(len(values_s1_s2))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]  # New set of positions for average bars

plt.bar(r1, values_s1_s2, width=barWidth, label='S1 → S2')
plt.bar(r2, values_s1_s3, width=barWidth, label='S1 → S3')
plt.bar(r3, values_s2_s1, width=barWidth, label='S2 → S1')
plt.bar(r4, values_s2_s3, width=barWidth, label='S2 → S3')
plt.bar(r5, values_s3_s1, width=barWidth, label='S3 → S1')
plt.bar(r6, values_s3_s2, width=barWidth, label='S3 → S2')
plt.bar(r7, averages, width=barWidth, color='gray', alpha=0.7, label='Average')  # Bar for average values

plt.xlabel('Methods', fontweight='bold')
plt.xticks([r + 3*barWidth for r in range(len(values_s1_s2))], methods)  # Adjust x-tick positions
plt.ylabel('Accuracy (%)')
plt.title('OPPT Dataset Results')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()

plt.savefig('oppt_results_with_avg.png', dpi=300)
plt.show()
