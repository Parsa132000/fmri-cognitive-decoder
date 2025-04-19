# 🧠 fMRI Cognitive State Classifier

A powerful end-to-end machine learning pipeline for classifying cognitive states from functional MRI (fMRI) data using PCA, SVM, and optional deep learning models. Built with neuroimaging libraries like Nilearn, MNE, NiBabel, PyTorch, and visualized in a Streamlit dashboard.

---

## 🚀 Features

- 🧠 Analyze real neuroimaging data (Haxby dataset)
- 📊 Dimensionality reduction (PCA)
- 🤖 Classification (SVM, optional: Deep Learning)
- 🕸️ Functional brain connectivity (optional)
- 🌍 Interactive Streamlit dashboard
- 📈 Brain activation visualizations (PCA components → brain maps)

---

## 📁 Project Structure

| Folder | Purpose |
|--------|---------|
| `notebooks/` | Colab & Jupyter Notebooks for interactive exploration |
| `src/` | Clean Python modules for preprocessing, training, and visualization |
| `streamlit_app/` | Streamlit web dashboard |
| `results/` | Output images like confusion matrices and brain maps |
| `data/` | Placeholder or data fetch script (public datasets only) |

---

## 🧰 Tech Stack

- `nilearn`, `nibabel`, `mne`
- `scikit-learn`, `matplotlib`, `seaborn`
- `PyTorch` (optional)
- `Streamlit` (dashboard)
- `pandas`, `numpy`

---

## 🔬 Use Case

Classify which kind of stimulus (e.g., face, house, cat) a subject is observing based on their brain activity recorded by fMRI. This can be extended to applications in:

- Cognitive neuroscience
- BCI (brain-computer interfaces)
- Mental state decoding
- Early diagnosis of neurological disorders

---

## 🏁 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fmri-cognitive-state-classifier.git
   cd fmri-cognitive-state-classifier
