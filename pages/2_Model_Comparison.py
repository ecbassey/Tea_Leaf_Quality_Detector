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

## colors: #c98967 #90A244 #7A7940 2f2f2
# -------------------------------
st.markdown("""
<style>
.stApp {
   
  background: linear-gradient(#134e5e, #71b280, #8AC399);
            
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)


st.markdown(
    "<h2 style='color:#ffffff;'>Model Performance</h2>",
    unsafe_allow_html=True
)

# st.write(
#     "Evaluate the performance of the trained deep learning models."
# )

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/models_accuracy3.png", caption="Model Performance", use_container_width=True)

with col2:
    st.text("")


st.markdown("---")


st.markdown(
    "<h4><b>Custom CNN Model</b></h4>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image(
        "images/accuracy_curve.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )

with col2:
    st.image(
        "images/loss_curve.png",
        caption="📉 CNN Loss Curve",
        use_container_width=True
    )


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/matrix.png", caption="Confusion Matrix", use_container_width=True)

with col2:
    st.image("images/classification.png", caption="CNN Classification Report", use_container_width=True)

#------------------------------------

st.markdown(
    "<h4><b>MobileNetV2 Model</b></h4>",
    unsafe_allow_html=True
)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image(
        "images/accuracy_curve_mobile.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )

with col2:
    st.image(
        "images/loss_curve_mobile.png",
        caption="📉 CNN Loss Curve",
        use_container_width=True
    )


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/matrix_mobile.png", caption="Confusion Matrix", use_container_width=True)

with col2:
    # st.text("")
    st.image("images/classification_mobile.png", caption="MobileNetV2 Classification Report", use_container_width=True)

#===========================

st.markdown(
    "<h4><b>Matrix</b></h4>",
    unsafe_allow_html=True
)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/matrix_eff.png", caption="Confusion Matrix", use_container_width=True)

with col2:
    st.image("images/matrix_resnet.png", caption="Confusion Matrix", use_container_width=True)