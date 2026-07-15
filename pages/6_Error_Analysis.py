import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import time

## colors: #c98967 #90A244 #7A7940 2f2f2
# st.markdown("""
# <style>
# .stApp {
#     background-color: #f2f2f2;
# }
# </style>
# """, unsafe_allow_html=True)
# -------------------------------
st.markdown("""
<style>
.stApp {
   
  background: linear-gradient(#134e5e, #71b280, #8AC399);
            
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# -------------------------------

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.markdown(
    "<h2 style='color:#ffffff;'>Error Analysis</h2>",
    unsafe_allow_html=True
)




st.markdown("------")
st.markdown("""
<style>
/* Label */
[data-testid="stMetricLabel"] {
    font-size:24px;
    color:#000000;
    font-weight:bold;
}

/* Value */
[data-testid="stMetricValue"] {
    font-size:30px;
    color:#000000;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h3 style='color:#ffffff;'>Misclassified Images</h3>",
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="
        background-color:#ffffff; 
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h4>19%</h4>
        <h3>Good Quality</h3>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h4>42%</h4>
        <h3>Average Quality</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h4>23%</h4>
        <h3>Not Good for Tea</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h4>1%</h4>
        <h3>Diseased Leaf</h3>
    </div>
    """, unsafe_allow_html=True)
st.text("")




col1, col2, col3 = st.columns(3)

col1.metric("Misclassified images", "85")

col2.metric("Average confidence (Correct)", "87%")

col3.metric("Average confidence (Wrong)", "65%")

#st.markdown("---")



#Misclassified images


st.image("images/misclassified.png",  use_container_width=True)

st.markdown(
    "<h4 style='color:#000000;'>Confidence Analysis</h4>",
    unsafe_allow_html=True
)

st.image("images/confidence_chart.png",  use_container_width=True)

st.markdown(
    "<h4 style='color:#000000;'>Confidence by class</h4>",
    unsafe_allow_html=True
)
st.image("images/confidence_class2.png",  use_container_width=True)



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="
        background-color:#ffffff; 
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h5>High confidence</h5>
        <h3>Good Quality</h3>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h5>Low confidence</h5>
        <h3>Average Quality</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h5>High confidence</h5>
        <h3>Not Good for Tea</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h5>Very high confidence</h5>
        <h3>Diseased Leaf</h3>
    </div>
    """, unsafe_allow_html=True)
st.text("")






st.markdown("---")

st.markdown("""
<div style="
    background-color:#1b1f24;
    padding:25px;
    border-radius:12px;
    border:1px solid #30363d;
">

<h2 style="
    color:#2dd4bf;
    font-size:24px;
    margin-bottom:15px;
">
Suggested Improvements
</h2>


<ul style="
    color:#d1d5db;
    font-size:21px;
    line-height:1.9;
">
    <li>Collect more training images.</li>
    <li>Increase the diversity of images (lighting, backgrounds, and angles).</li>
    <li>Apply stronger or more targeted data augmentation.</li>
    <li>Fine-tune more layers of MobileNetV2.</li>
    <li>Experiment with different learning rates (e.g., 0.0001–0.0003).</li>
    <li>Use a higher image resolution if computationally feasible.</li>
    <li>Apply image segmentation to isolate the tea leaf from the background.</li>
</ul>

</div>
""", unsafe_allow_html=True)

st.markdown("---")


