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

st.write("""
    Welcome to the Hierarchical Clustering Demonstrator. This app allows you to visualize 
    hierarchical clustering using different linkage methods. Adjust the sliders and dropdown in the sidebar 
    to change the parameters and observe how the dendrogram changes.
""")

# Sidebar settings with explanations
st.sidebar.header("Hierarchical Clustering Settings")
st.sidebar.markdown("""
    **Number of Samples**  
    Choose how many data points to generate for clustering.
""")
num_samples = st.sidebar.slider("Number of Samples", 10, 200, 100)

st.sidebar.markdown("""
    **Number of Features**  
    Select the number of features (dimensions) for each data point.
""")
num_features = st.sidebar.slider("Number of Features", 2, 5, 2)

st.sidebar.markdown("""
    **Linkage Method**  
    Select the method used to calculate the distance between clusters.
""")
linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ("ward", "complete", "average", "single")
)

st.sidebar.markdown("""
    **Cut-off Distance**  
    Set a threshold to define the maximum distance between clusters. Clusters above this 
    distance will not be joined.
""")
max_d = st.sidebar.slider("Cut-off Distance", 0.0, 200.0, 25.0)

st.markdown("""
    **Data Generation**  
    The dataset is generated with a specified number of samples and features, 
    which are randomly distributed in space. 
""")

# Generate synthetic data
X, _ = make_blobs(n_samples=num_samples, n_features=num_features, random_state=42)

# Perform hierarchical clustering
Z = linkage(X, method=linkage_method)

# Show dendrogram with explanations
st.subheader(f"Dendrogram ({linkage_method} linkage)")
st.write("""
    A dendrogram visualizes the process of hierarchical clustering. The y-axis represents 
    the distance between clusters being merged, while the x-axis represents the individual 
    data points or merged clusters.
""")
plot_dendrogram(Z)

# Show truncated dendrogram with explanations
st.subheader(f"Truncated Dendrogram ({linkage_method} linkage) with Cut-off Distance")
st.write("""
    The truncated dendrogram shows the last few merges of the hierarchical clustering process. 
    The horizontal line represents the cut-off distance used to define clusters.
""")
plot_dendrogram(Z, max_d=max_d)

# Determine and report the number of clusters
clusters = fcluster(Z, max_d, criterion='distance')
st.write(f'Number of clusters formed at cut-off distance {max_d}: {len(set(clusters))}')
