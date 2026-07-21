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
    "<h2 style='color:#ffffff;'>Grad-CAM Explainability</h2>",
    unsafe_allow_html=True
)



st.markdown("""
<style>

.gradcam-card{
    background:#1b1f24;
    border:1px solid #30363d;
    border-radius:15px;
    padding:25px;
    margin-top:20px;
}

.gradcam-title{
    color:#2dd4bf;
    font-size:24px;
    font-weight:bold;
    margin-bottom:15px;
}

.gradcam-text{
    color:#d1d5db;
    font-size:20px;
    line-height:1.4;
    text-align:justify;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gradcam-card">

<div class="gradcam-title">
How Grad-CAM Works
</div>

<div class="gradcam-text">
Gradient-weighted Class Activation Mapping (Grad-CAM) is an explainable AI
technique that highlights the regions of a tea leaf image that most influenced
the CNN's prediction. By analyzing the gradients flowing into the final
convolutional layer, Grad-CAM generates a heatmap that identifies the most
important features used for classification. This enables users to understand
whether the model is focusing on meaningful leaf characteristics such as
veins, discoloration, disease spots, or damaged edges when predicting one of
the four tea leaf quality classes.
</div>

</div>
""", unsafe_allow_html=True)

st.text("")
st.image("images/grad-cam1_1.png",  use_container_width=True)
st.image("images/grad-cam2_1.png",  use_container_width=True)
st.image("images/grad-cam13_1.png",  use_container_width=True)
st.image("images/grad-cam14_1.png",  use_container_width=True)
st.image("images/grad-cam5.png",  use_container_width=True)
st.image("images/grad-cam6_1.png",  use_container_width=True)
st.image("images/grad-cam7.png",  use_container_width=True)
st.image("images/grad-cam8_1.png",  use_container_width=True)
st.image("images/grad-cam9_1.png",  use_container_width=True)
st.image("images/grad-cam12_1.png",  use_container_width=True)
st.image("images/grad-cam11_1.png",  use_container_width=True)
st.image("images/grad-cam15_1.png",  use_container_width=True)
