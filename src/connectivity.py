import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting


def compute_correlation_matrix(fmri_data, kind='correlation'):
    """
    Computes the functional connectivity matrix.

    Parameters:
    - fmri_data: 2D array (time points × voxels or regions)
    - kind: str, one of 'correlation', 'partial correlation', or 'tangent'

    Returns:
    - connectivity_matrix: 2D array (regions × regions)
    """
    measure = ConnectivityMeasure(kind=kind)
    conn_matrix = measure.fit_transform([fmri_data])[0]
    return conn_matrix


def plot_connectivity_matrix(conn_matrix, title="Functional Connectivity", labels=None):
    """
    Plots a heatmap of the functional connectivity matrix.

    Parameters:
    - conn_matrix: 2D array
    - title: str
    - labels: list of region names (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conn_matrix, cmap='coolwarm', square=True,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.tight_layout()
    plt.show()


def plot_connectome_on_brain(conn_matrix, coords, threshold=0.2, title="Connectome Visualization"):
    """
    Visualizes the connectome over brain coordinates.

    Parameters:
    - conn_matrix: 2D array (connectivity matrix)
    - coords: list of (x, y, z) tuples or array of region coordinates
    - threshold: float, minimum absolute connectivity strength to display
    - title: str
    """
    plotting.plot_connectome(conn_matrix, coords, edge_threshold=threshold,
                             node_size=20, title=title)
