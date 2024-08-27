import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have arrays for ground truth and predicted depths
ground_truth_depths = np.random.rand(1000)
predicted_depths = ground_truth_depths + np.random.normal(0, 0.1, 1000)

# 1. Depth Error Histogram
depth_errors = predicted_depths - ground_truth_depths
plt.figure()
plt.hist(depth_errors, bins=50, alpha=0.7)
plt.title('Depth Error Histogram')
plt.xlabel('Depth Error')
plt.ylabel('Frequency')
plt.show()

# 2. Scatter Plot
plt.figure()
plt.scatter(ground_truth_depths, predicted_depths, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
plt.title('Ground Truth vs Predicted Depths')
plt.xlabel('Ground Truth Depth')
plt.ylabel('Predicted Depth')
plt.show()

# 4. Error Heatmap (Example with random data)
error_heatmap = np.abs(np.random.rand(100, 100) - np.random.rand(100, 100))
plt.figure()
sns.heatmap(error_heatmap, cmap='viridis')
plt.title('Error Heatmap')
plt.show()

# 6. Cumulative Error Distribution
sorted_errors = np.sort(np.abs(depth_errors))
cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
plt.figure()
plt.plot(sorted_errors, cdf)
plt.title('Cumulative Error Distribution')
plt.xlabel('Absolute Error')
plt.ylabel('Cumulative Proportion')
plt.show()

# 7. Quantitative Metrics Bar Chart
metrics = {'RMSE': np.sqrt(np.mean(depth_errors**2)), 'MAE': np.mean(np.abs(depth_errors))}
plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.title('Quantitative Metrics')
plt.ylabel('Value')
plt.show()
