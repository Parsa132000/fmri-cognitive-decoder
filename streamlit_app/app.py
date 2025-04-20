# Placeholder for app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from nilearn.datasets import fetch_haxby
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map

st.set_page_config(page_title="🧠 fMRI Classifier", layout="wide")

st.title("🧠 fMRI Cognitive State Classifier")
st.markdown("Classify cognitive states based on fMRI activity and visualize brain maps using PCA and SVM.")

# Load dataset
@st.cache_data
def load_data():
    haxby = fetch_haxby()
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    condition_mask = labels["labels"] != "rest"
    labels_filtered = labels[condition_mask]
    fmri_img = index_img(haxby.func[0], condition_mask)
    masker = NiftiMasker(mask_img=haxby.mask_vt[0], standardize=True)
    fmri_masked = masker.fit_transform(fmri_img)
    return fmri_masked, labels_filtered["labels"].values, masker

X, y, masker = load_data()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Sidebar
st.sidebar.title("🛠 Settings")
n_components = st.sidebar.slider("PCA Components", 10, 150, 100)
selected_component = st.sidebar.slider("Brain Component View", 0, n_components - 1, 0)
classifier_type = st.sidebar.selectbox("Classifier", ["SVM"])

# PCA
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)

# Train SVM
clf = SVC(kernel="linear")
clf.fit(X_reduced, y_encoded)
y_pred = clf.predict(X_reduced)

# Metrics
accuracy = np.mean(y_pred == y_encoded)
st.subheader("📈 Classification Accuracy")
st.success(f"{accuracy:.2%}")

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_encoded, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig_cm)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_encoded, y_pred, target_names=le.classes_)
st.text(report)

# Brain Visualization
st.subheader(f"🧠 Brain Map: PCA Component {selected_component + 1}")
component_img = masker.inverse_transform(pca.components_[selected_component])
plot_stat_map(component_img, display_mode="z", cut_coords=5, threshold=1.0,
              title=f"PCA Component {selected_component + 1}")
