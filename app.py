import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd

## colors: #c98967 #90A244 #7A7940 2f2f2  background-color: #000000;
# background: linear-gradient(135deg, #74ebd5, #9face6);
# background: linear-gradient(135deg, #134e5e, #71b280);
#background: linear-gradient(135deg, #134e5e, #71b280);

st.markdown("""
<style>
.stApp {
   
  background: linear-gradient(#134e5e, #71b280, #8AC399);
            
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.body-text {
    color: #166534;
    font-size: 20px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# st.markdown("""
# <style>
# .hero{
#     background: linear-gradient(135deg,#3b82f6,#9333ea);
#     border-radius:20px;
#     padding:60px;
#     color:white;
#     backdrop-filter: blur(15px);
#     box-shadow:0 8px 32px rgba(31,38,135,0.3);
# }
# </style>
#             """, unsafe_allow_html=True)

# ------------------------------------
# PAGE CONFIGURATION
# ------------------------------------
st.set_page_config(
    page_title="Tea Leaf Quality Detection",
    page_icon="🍃",
    layout="wide"
)

# ------------------------------------
# TITLE
# ------------------------------------

# st.title("Tea Leaf Quality Detection and Intelligence System")

# st.markdown("---")
st.markdown(
    "<h1 style='color:#ffffff;'>Tea Leaf Quality Detection and Intelligence System</h1>",
    unsafe_allow_html=True
)
# ------------------------------------
# PROJECT OVERVIEW
# ------------------------------------

st.markdown(
    "<h3 style='color:#000000;'>PROJECT OVERVIEW</h3>",
    unsafe_allow_html=True
)


st.markdown("""
<div style="
    color: #000000;
    font-size: 20px;
    line-height: 1.7;
">

The Tea Leaf Quality Detection and Intelligence System is an Artificial Intelligence
application developed to automatically classify tea leaves into four quality categories
using Deep Learning and Computer Vision.

<br><br>

The system analyzes images of tea leaves and predicts whether the leaf belongs to one
of the following classes:

</div>
""", unsafe_allow_html=True)

# ------------------------------------
# classes
# ------------------------------------
#st.markdown("## 🍃 Tea Leaf Quality Classes")
# #d4edda

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="
        background-color:#ffffff; 
        padding:10px;
        border-radius:12px;
        text-align:center;
        border:2px solid #28a745;">
        <h4>CLASS 1</h4>
        <h3>Premium Quality</h3>
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
        <h4>CLASS 2</h4>
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
        <h4>CLASS 3</h4>
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
        <h4>CLASS 4</h4>
        <h3>Diseased</h3>
    </div>
    """, unsafe_allow_html=True)
st.text("")
st.markdown("The objective is to assist tea farmers and quality inspectors by providing fast, consistent, and intelligent quality assessment.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.image("tea_leaf.jpg",  width=400)

with col2:
    st.image("tea_leaf2.jpg",  width=400)





# ------------------------------------
# DATASET SUMMARY
# ------------------------------------

st.markdown(
    "<h3 style='color:#ffffff;'>DATASET SUMMARY</h3>",
    unsafe_allow_html=True
)

dataset_path = "tea_dataset_resized"

classes = sorted(os.listdir(dataset_path))

num_classes = len(classes)

image_count = 0

class_counts = {}

for c in classes:

    folder = os.path.join(dataset_path, c)

    count = len(os.listdir(folder))

    class_counts[c] = count

    image_count += count

# ------------------------------------
# METRICS
# ------------------------------------
st.markdown("""
<style>

/* Metric label (e.g., Number of Classes) */
[data-testid="stMetricLabel"] {
    font-size: 20px;
    color: #ffffff;
    font-weight: 600;
}

/* Metric value (e.g., 4, 800, 224 × 224) */
[data-testid="stMetricValue"] {
    font-size: 36px;
    color: #1F2937;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

col1.metric("Number of Classes", num_classes)

col2.metric("Total Images", image_count)

col3.metric("Image Size", "224 × 224")

#st.markdown("---")



# ------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------

st.markdown(
    "<h3 style='color:#ffffff;'>Dataset Distribution</h3>",
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="
        background-color:#fff3cd;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #fd7e14;">
        <h4>Average Quality</h4>
        <h2>508</h2>
        <p>Images</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background-color:#fff3cd;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #fd7e14;">
        <h4>Diseased</h4>
        <h2>623</h2>
        <p>Images</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background-color:#fff3cd;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #fd7e14;">
        <h4>Not Good</h4>
        <h2>523</h2>
        <p>Images</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
        background-color:#fff3cd;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #fd7e14;">
        <h4>High Quality</h4>
        <h2>631</h2>
        <p>Images</p>
    </div>
    """, unsafe_allow_html=True)    




# ------------------------------------
# MODEL PERFORMANCE
# ------------------------------------

st.markdown(
    "<h3 style='color:#ffffff;'><br>Current Best Model</h3>",
    unsafe_allow_html=True
)

best_model = "Custom CNN"
best_accuracy = "82.6 %"
col1, col2 = st.columns(2)
col1.metric("Best Performing Model", best_model)
col2.metric("Validation Accuracy", best_accuracy)


#st.markdown("---")


st.markdown("""
<div style="
    background-color: #043927;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    border-left: 5px solid #ffffff;
">
            
The best-performing model is selected after comparing: \n
<strong> \n
• Custom CNN \n
• MobileNetV2 \n
• ResNet50 \n
• EfficientNetB0 \n
</strong>
            
</div>
""", unsafe_allow_html=True)
# ------------------------------------
# PROJECT OBJECTIVES
# ------------------------------------

st.markdown(
    "<h3 style='color:#ffffff;'><br>Project Objectives</h3>",
    unsafe_allow_html=True
)


st.markdown("""
<div style="
    color: #000000;
    font-size: 20px;
    line-height: 1.8;
">
<ul>
    <li>Develop an intelligent tea leaf classification system.</li>
    <li>Detect diseased tea leaves automatically.</li>
    <li>Compare multiple Deep Learning architectures.</li>
    <li>Explain predictions using Grad-CAM.</li>
    <li>Build an interactive analytics dashboard.</li>
    <li>Improve consistency in tea quality assessment.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ------------------------------------
# TECHNOLOGIES
# ------------------------------------

st.markdown(
    "<h3 style='color:#ffffff;'><br>Technologies Used</h3>",
    unsafe_allow_html=True
)

st.markdown("""
<style>

.tech-box {
    background-color: #E8F5E9;
    color: #1F2937;
    font-size: 20px;
    font-weight: 500;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #043927;
    line-height: 1.8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="tech-box">
    Python<br>
    TensorFlow<br>
    Keras<br>
    OpenCV
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tech-box">
    NumPy<br>
    Pandas<br>
    Matplotlib<br>
    Scikit-Learn
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tech-box">
    Streamlit<br>
    Grad-CAM<br>
    Transfer Learning<br>
    CNN
    </div>
    """, unsafe_allow_html=True)



st.markdown("---")

# ------------------------------------
# Data Sources
# ------------------------------------

st.markdown(
    """
    <p style="color:#ffffff; font-size:20px;">
        The images below show one example from each tea leaf quality class.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    "<h3 style='color:#ffffff;'><br>Data Sources</h3>",
    unsafe_allow_html=True
)

st.markdown("""
            
- Mendeley Data : TeaLeafAgeQuality: Age-Stratified Tea Leaf Quality Classification Dataset.
        Published: 2 January 2024
|
Version 1
|
DOI:
10.17632/7t964jmmy3.1    https://data.mendeley.com/datasets/7t964jmmy3/1 
            

- Roboflow: Tea leaf diseases.
            https://universe.roboflow.com/tea-leaf-diseases/tea-leaf-diseases-7eqxh

             
""")

st.markdown("---")


# ------------------------------------
# FOOTER
# ------------------------------------

st.caption("Tea Leaf Quality Detection and Intelligence System | Final Year AI Project")

