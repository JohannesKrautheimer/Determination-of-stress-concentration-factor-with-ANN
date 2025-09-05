from time import time
import matplotlib.pyplot as plt

def plot_before_and_after_pointnet(points_before, points_after, title=""):
    x_values_before, y_values_before = points_before[:, 0], points_before[:, 1]
    x_values_after, y_values_after = points_after[:, 0], points_after[:, 1]

    plot_points(x_values_before, y_values_before, x_values_after, y_values_after, title=title)

    
def plot_before_and_after_lasernet(points_before, points_after, title=""):
    x_values_before, y_values_before = points_before[0, :], points_before[1, :]
    x_values_after, y_values_after = points_after[0, :], points_after[1, :]

    plot_points(x_values_before, y_values_before, x_values_after, y_values_after, title=title)

def plot_points(x_before, y_before, x_after, y_after, title=""):
    plt.figure(figsize=(10, 5))
    # Plot the points before
    plt.subplot(1, 2, 1)
    plt.scatter(x_before, y_before, c='blue', label=f'Before Augmentation: {title}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Plot the points after
    plt.subplot(1, 2, 2)
    plt.scatter(x_after, y_after, c='red', label=f'After Augmentation: {title}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.tight_layout()
    plt.show()