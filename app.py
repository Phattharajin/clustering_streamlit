import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_blobs

# Load model
with open('kmeans_model.pkl', 'rb') as f:
   loaded_model = pickle.load(f)

st.set_page_config(page_title="k-means Clustering App", layout="centered")

st.title("k-means Clustering Visualizer")

st.subheader("Example data visualization")
st.markdown("This demo uses example data")

# Generate example data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=42)

# Predict cluster labels using the loaded k-means model
y_kmeans = loaded_model.predict(X)

# Create scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Mark the centroids
centers = loaded_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')

# Display plot
st.pyplot(plt)
