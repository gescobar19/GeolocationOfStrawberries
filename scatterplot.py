import matplotlib.pyplot as plt

# Ground truth depths
ground_truth = [73, 70, 72, 70, 69, 71, 73, 73]

# Depths estimated by different models
depths_vits = [68.00, 68.00, 68.00, 68.06, 69.37, 71.18, 68.00, 68.00]
depths_vitb = [69.73, 68.00, 68.68, 70.29, 71.28, 70.74, 68.00, 68.00]
depths_vitl = [68.00, 68.00, 68.00, 68.00, 68.00, 68.00, 68.00, 68.00]
depths_midas = [70.68, 70.60, 70.47, 70.75, 70.43, 70.57, 70.28, 70.95]

# Strawberry labels
strawberries = range(1, 9)

# Plotting the data
plt.figure(figsize=(12, 8))

plt.plot(strawberries, ground_truth, label='Ground Truth', color='black', marker='o')
plt.plot(strawberries, depths_vits, label='vits', color='red', marker='o')
plt.plot(strawberries, depths_vitb, label='vitb', color='blue', marker='o')
plt.plot(strawberries, depths_vitl, label='vitl', color='green', marker='o')
plt.plot(strawberries, depths_midas, label='midas', color='purple', marker='o')

plt.xlabel('Strawberry Number',fontsize=14)
plt.ylabel('Depth (cm)',fontsize=14)
plt.title('Depth Estimations by Different Models Compared to Ground Truth',fontsize=15)
plt.legend(fontsize=16)  # Increase the font size of the legend
plt.grid(True)

# Show the plot
plt.show()
