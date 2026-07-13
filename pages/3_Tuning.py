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
    
    background: linear-gradient(135deg, #134e5e, #71b280);
}
</style>
""", unsafe_allow_html=True)
# -------------------------------


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.markdown(
    "<h2 style='color:#ffffff;'>Hyper Parameter Tuning - (CNN)</h2>",
    unsafe_allow_html=True
)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        """
        <p style="
            color: #000000;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        ">
            Original Setting
        </p>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <p style="
            color: #000000;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        ">
            Hyper Tuned
        </p>
        """,
        unsafe_allow_html=True
    )




col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/aug.png", caption="Augmentation", use_container_width=True)

with col2:
    st.image("images/hyper_aug3.png", caption="Augmentation", use_container_width=True)



col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/cnn_build.png", caption="CNN Build", use_container_width=True)

with col2:
    st.image("images/hyper_cnn_build.png", caption="CNN Build", use_container_width=True)



col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/opt.png", caption="Optimizer", use_container_width=True)

with col2:
    st.image("images/hyper_opt2.png", caption="Optimizer", use_container_width=True)



col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("images/best_epoch.png", caption="Accuracy", use_container_width=True)

with col2:
    st.image("images/hyper_best_epoch2.png", caption="Accuracy", use_container_width=True)





col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image(
        "images/accuracy_curve_pre.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )

with col2:
    st.image(
        "images/accuracy_curve.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image(
        "images/loss_curve_pre.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )

with col2:
    st.image(
        "images/loss_curve.png",
        caption="📈 CNN Accuracy Curve",
        use_container_width=True
    )