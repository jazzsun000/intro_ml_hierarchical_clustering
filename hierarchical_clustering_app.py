# Required installations:
# pip install streamlit scikit-learn matplotlib scipy

import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import fcluster

# Function to plot dendrogram
def plot_dendrogram(Z, max_d=None):
    # Create the dendrogram
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    if max_d:
        plt.axhline(y=max_d, c='k')
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    plt.show()
    st.pyplot(fig)

st.title("Hierarchical Clustering Demonstrator")

# Sidebar settings
st.sidebar.header("Hierarchical Clustering Settings")
num_samples = st.sidebar.slider("Number of Samples", 10, 200, 100)
num_features = st.sidebar.slider("Number of Features", 2, 5, 2)
linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ("ward", "complete", "average", "single")
)

# Generate synthetic data
X, _ = make_blobs(n_samples=num_samples, n_features=num_features, random_state=42)

# Perform hierarchical clustering
Z = linkage(X, method=linkage_method)

# Show dendrogram
st.subheader(f"Dendrogram ({linkage_method} linkage)")
plot_dendrogram(Z)

# Allow user to set a cut-off distance to determine clusters
max_d = st.sidebar.slider("Cut-off Distance", 0.0, 200.0, 25.0)
st.subheader(f"Truncated Dendrogram ({linkage_method} linkage) with Cut-off Distance")
plot_dendrogram(Z, max_d=max_d)

# Determine and report the number of clusters as per cut-off distance
clusters = fcluster(Z, max_d, criterion='distance')
st.write(f'Number of clusters formed: {len(set(clusters))}')
