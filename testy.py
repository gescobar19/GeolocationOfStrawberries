import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix values
TP = 240
FP = 12
FN = 38
TN = 110  # This value needs to be calculated or provided

# Creating the confusion matrix
conf_matrix = np.array([[TP, FN],
                        [FP, TN]])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
