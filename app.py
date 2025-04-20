# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 00:30:43 2025

"""

import streamlit as st
import matplotlib.pyplot as plt
import pickle

# Load model
with open('kmeans_model.pkl', 'rb') as f:
   loaded_model = pickle.load(f)
   
st.set_page_config(page_title="k-means Clustering App", layout="centered")


st.title("k-means Clustering Visualizer")

st.subheader("Example data visualization")
st.markdown("This demo uses example data")

from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=42)
                
                  
y_kmeans = loaded_model.predict(X)              