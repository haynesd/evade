
from matplotlib import pyplot as plt


def visualize_results(model_names, accuracies, train_times, pred_times):
    """
    Visualize the results of model comparisons and save the plot to a file.
    Args:
        model_names (list): Names of the models.
        accuracies (list): Accuracy scores of the models.
        train_times (list): Training times for the models.
        pred_times (list): Prediction times for the models.
    """

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Accuracy bar chart
    ax[0].bar(model_names, accuracies, color='blue')
    ax[0].set_title("Accuracy")
    ax[0].set_ylabel("Accuracy")

    # Training time bar chart
    ax[1].bar(model_names, train_times, color='green')
    ax[1].set_title("Training Time")
    ax[1].set_ylabel("Time (seconds)")

    # Prediction time bar chart
    ax[1].bar(model_names, pred_times, color='red')
    ax[1].set_title("Prediction Time")
    ax[1].set_ylabel("Time (seconds)")

    plt.tight_layout()

    # Save the plot to a file (e.g., PNG format)
    plt.savefig('model_comparison.png')


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
