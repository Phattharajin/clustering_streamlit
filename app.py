import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_blobs

# Load pre-trained k-means model (make sure 'kmeans_model.pkl' is in your project folder or specify the correct path)
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set up Streamlit page configuration
st.set_page_config(page_title="k-means Clustering", layout="centered")

# Add title and subheader to the Streamlit app
st.title("k-means Clustering Visualizer by Phattharajin Joyjaroen")
st.subheader("This is an interactive visualization of k-means clustering with a pre-trained model.")

# Generate synthetic example data using make_blobs (this mimics your real dataset)
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict cluster labels using the loaded k-means model
y_kmeans = loaded_model.predict(X)

# Create a scatter plot of the clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.7)

# Mark the cluster centers
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='X')

# Add title and labels
ax.set_title('k-Means Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Add legend (for cluster colors)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# Show the plot in Streamlit
st.pyplot(fig)
