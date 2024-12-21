from matplotlib import pyplot as plt
import numpy as np


def visualize_results(model_names, accuracies, train_times, pred_times):
    """
    Visualize the results of model comparisons and save the plot to a file.
    Args:
        model_names (list): Names of the models.
        accuracies (list): Accuracy scores of the models.
        train_times (list): Training times for the models.
        pred_times (list): Prediction times for the models.
    """

    bar_width = 0.25  # Width of each bar
    r1 = np.arange(len(model_names))  # Position for Accuracy bars
    r2 = [x + bar_width for x in r1]  # Position for Training Time bars
    r3 = [x + bar_width for x in r2]  # Position for Prediction Time bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(r1, accuracies, color='blue', width=bar_width,
           edgecolor='grey', label='Accuracy')
    ax.bar(r2, train_times, color='green', width=bar_width,
           edgecolor='grey', label='Training Time')
    ax.bar(r3, pred_times, color='red', width=bar_width,
           edgecolor='grey', label='Prediction Time')

    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Scores/Time (seconds)', fontweight='bold')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks([r + bar_width for r in range(len(model_names))])
    ax.set_xticklabels(model_names)

    # Add legend
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('model_comparison.png')
    plt.show()


# Example usage
model_names = ['HDBSCAN']
accuracies = [1.0]
train_times = [0.6]
pred_times = [0.2]
visualize_results(model_names, accuracies, train_times, pred_times)


def plot_hdbscan_results(X, cluster_labels):
    """
    Visualize the HDBSCAN clustering results.
    """
    plt.figure(figsize=(10, 6))
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            # Noise points
            color = 'k'
            label_name = "Noise"
        else:
            color = plt.cm.jet(float(label) / max(unique_labels))
            label_name = f"Cluster {label}"
        plt.scatter(X[cluster_labels == label, 0], X[cluster_labels ==
                    label, 1], c=color, label=label_name, alpha=0.6)
    plt.legend()
    plt.title("HDBSCAN Clustering Results")
    plt.show()
