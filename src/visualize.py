# Placeholder for visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from nilearn.plotting import plot_stat_map


def plot_confusion(y_true, y_pred, class_labels, title="Confusion Matrix"):
    """
    Plots a labeled confusion matrix using Seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, target_names):
    """
    Prints precision, recall, f1-score report.
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def show_pca_brain_map(component_array, masker, component_idx=0, threshold=1.0, cut_coords=5):
    """
    Reconstructs and displays a PCA component in brain space.
    """
    img = masker.inverse_transform(component_array[component_idx])
    plot_stat_map(img, title=f'PCA Component {component_idx+1}',
                  display_mode='z', cut_coords=cut_coords, threshold=threshold)
