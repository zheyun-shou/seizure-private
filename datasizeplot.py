# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import make_interp_spline

# # Your data
# datasize = np.array([0.6, 0.4, 0.3, 0.5, 0.2])
# epoch_wise_test_acc = np.array([0.82, 0.77, 0.74, 0.77, 0.44])
# epoch_wise_f1 = np.array([0.64, 0.602, 0.629, 0.674, 0.249])
# event_wise_f1 = np.array([0.925, 0.868, 0.935, 0.912, 0.8])

# # Sort by datasize (to make the line smooth and correct)
# sorted_indices = np.argsort(datasize)
# datasize = datasize[sorted_indices]
# epoch_wise_test_acc = epoch_wise_test_acc[sorted_indices]
# epoch_wise_f1 = epoch_wise_f1[sorted_indices]
# event_wise_f1 = event_wise_f1[sorted_indices]

# # Smooth interpolation
# xnew = np.linspace(datasize.min(), datasize.max(), 300)

# spl1 = make_interp_spline(datasize, epoch_wise_test_acc, k=2)
# spl2 = make_interp_spline(datasize, epoch_wise_f1, k=2)
# spl3 = make_interp_spline(datasize, event_wise_f1, k=2)

# plt.figure(figsize=(8, 6))
# plt.plot(xnew, spl1(xnew), label='Epoch-wise Test Accuracy', color='blue')
# plt.plot(xnew, spl2(xnew), label='Epoch-wise F1', color='green')
# plt.plot(xnew, spl3(xnew), label='Event-wise F1', color='red')

# plt.scatter(datasize, epoch_wise_test_acc, color='blue')
# plt.scatter(datasize, epoch_wise_f1, color='green')
# plt.scatter(datasize, event_wise_f1, color='red')

# plt.xlabel('Datasize')
# plt.ylabel('Score')
# plt.title('Performance Metrics vs Datasize')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Your data
# datasize = np.array([0.6, 0.4, 0.3, 0.5, 0.2])
# epoch_wise_test_acc = np.array([0.82, 0.77, 0.74, 0.77, 0.44])
# epoch_wise_f1 = np.array([0.64, 0.602, 0.629, 0.674, 0.249])
# event_wise_f1 = np.array([0.925, 0.868, 0.935, 0.912, 0.8])
datasize = np.array([0.3, 0.4, 0.6, 0.7, 0.5, 0.8])
epoch_wise_test_acc = np.array([0.71, 0.78, 0.73, 0.75, 0.79, 0.82])
epoch_wise_f1 = np.array([0.642, 0.558, 0.614, 0.529, 0.55, 0.674])
event_wise_f1 = np.array([0.94, 0.931, 0.897, 0.834, 0.864, 0.888])

# Sort by datasize
sorted_indices = np.argsort(datasize)
datasize = datasize[sorted_indices]
epoch_wise_test_acc = epoch_wise_test_acc[sorted_indices]
epoch_wise_f1 = epoch_wise_f1[sorted_indices]
event_wise_f1 = event_wise_f1[sorted_indices]

# Plot straight lines and points
plt.figure(figsize=(8, 6))
plt.plot(datasize, epoch_wise_test_acc, label='Epoch-wise Test Accuracy', color='blue', marker='o')
plt.plot(datasize, epoch_wise_f1, label='Epoch-wise F1', color='green', marker='o')
plt.plot(datasize, event_wise_f1, label='Event-wise F1', color='red', marker='o')

plt.xlabel('Datasize')
plt.ylabel('Score')
plt.title('Performance Metrics vs Datasize')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
