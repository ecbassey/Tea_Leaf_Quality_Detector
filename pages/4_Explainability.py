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
    "<h2 style='color:#ffffff;'>Explainability</h2>",
    unsafe_allow_html=True
)


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




st.markdown("""
<div style="
    background-color: #043927;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    border-left: 5px solid #ffffff;
">
            
erorr Analysis: \n
<strong> \n
• Confusion Matrix \n
• Classification Report \n
• Misclassified images \n
• Grad-Cam \n
• Confidence Analysis \n
</strong>
            
</div>
""", unsafe_allow_html=True)

st.markdown("""
Possible Causes of Errors

Discuss likely reasons:

Small dataset \n
Class imbalance \n
Similar disease appearance \n
Low image quality \n
Shadows \n
Different camera angles \n
Background clutter \n
Disease symptoms overlap
            """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
Suggested Improvements

Explain how the model could be improved.

Examples:

Collect more training images.\n
Increase the diversity of images (lighting, backgrounds, angles).\n
Apply stronger or more targeted data augmentation.\n
Fine-tune more layers of MobileNet.\n
Experiment with learning rates (e.g., 0.0001–0.0003).\n
Use higher image resolution if computationally feasible.\n
Apply segmentation to isolate the leaf from the background.\n \n \n
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
Example Error Analysis Paragraph

The confusion matrix indicates that the model performs well on Healthy and 
            Red Leaf Spot leaves but struggles to distinguish Brown Blight from White Spot. 
            Examination of misclassified images revealed that these diseases exhibit similar
             discoloration patterns and lesion shapes. Grad-CAM visualizations showed that, 
            in several incorrect predictions, the model focused on leaf edges or background 
            regions instead of the diseased areas, suggesting that irrelevant features 
            influenced the prediction. Additionally, most incorrect classifications had 
            relatively low confidence scores, indicating model uncertainty. Future improvements 
            could include collecting more images for the confusing classes, 
            improving background removal, applying more targeted data augmentation, 
            and further fine-tuning the pretrained MobileNet model.

""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
How to discuss the results in your report

The model achieved an average confidence of 96.3% for correctly classified images, 
            compared to 68.1% for misclassified images. Most incorrect predictions had 
            confidence below 80%, suggesting that the model was generally uncertain when making errors.
             However, a small number of high-confidence misclassifications were observed, indicating that certain disease classes have similar visual characteristics and are difficult to distinguish. This suggests that additional training data or improved feature extraction could help reduce these errors.
""", unsafe_allow_html=True)