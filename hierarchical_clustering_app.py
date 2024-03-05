import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import fcluster
import numpy as np

st.title("Hierarchical Clustering Demonstrator")

st.write("""
    Welcome to the Hierarchical Clustering Demonstrator. This app allows you to visualize 
    hierarchical clustering using different linkage methods. Adjust the sliders and dropdown in the sidebar 
    to change the parameters and observe how the dendrogram and clustering change.
""")

# Sidebar settings with explanations
st.sidebar.header("Hierarchical Clustering Settings")
num_samples = st.sidebar.slider("Number of Samples", 10, 200, 100, help="Choose how many data points to generate for clustering.")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ("ward", "complete", "average", "single"),
    help="Select the method used to calculate the distance between clusters."
)

max_d = st.sidebar.slider("Cut-off Distance", 0.0, 200.0, 25.0, help="Set a threshold to define the maximum distance between clusters. Clusters above this distance will not be joined.")

# Generate synthetic data with 2 features
X, _ = make_blobs(n_samples=num_samples, centers=4, n_features=2, random_state=42)

# Perform hierarchical clustering
Z = linkage(X, method=linkage_method)

# Plotting function for dendrogram
def plot_dendrogram(Z, max_d=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
    if max_d:
        ax.axhline(y=max_d, c='k', linestyle='--')
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

# Plot the dendrogram
st.subheader(f"Dendrogram ({linkage_method} linkage)")
plot_dendrogram(Z)

# Truncated dendrogram
st.subheader(f"Truncated Dendrogram ({linkage_method} linkage) with Cut-off Distance")
plot_dendrogram(Z, max_d=max_d)

# Determine clusters based on cut-off distance
clusters = fcluster(Z, max_d, criterion='distance')

# Scatter plot to visualize clusters
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.title('Clustered Data Points')
st.pyplot(fig)

st.write(f'Number of clusters formed at cut-off distance {max_d}: {len(set(clusters))}')
