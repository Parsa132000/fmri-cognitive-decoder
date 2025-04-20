from nilearn.datasets import fetch_haxby
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
import pandas as pd


def load_and_preprocess_data(mask_type="vt", standardize=True):
    """
    Load and preprocess the Haxby fMRI dataset.

    Parameters:
    - mask_type: str, which mask to use ('vt' = ventral temporal)
    - standardize: bool, whether to standardize the data

    Returns:
    - X: voxel time series (numpy array)
    - y: labels (stimulus category)
    - masker: fitted NiftiMasker for inverse transforms
    """
    haxby = fetch_haxby()
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    condition_mask = labels["labels"] != "rest"
    labels_filtered = labels[condition_mask]

    # Load volumes matching the condition
    fmri_img = index_img(haxby.func[0], condition_mask)

    # Choose appropriate mask
    mask_img = haxby.mask_vt[0] if mask_type == "vt" else haxby.mask_face[0]

    # Apply mask
    masker = NiftiMasker(mask_img=mask_img, standardize=standardize)
    X = masker.fit_transform(fmri_img)

    return X, labels_filtered["labels"].values, masker
