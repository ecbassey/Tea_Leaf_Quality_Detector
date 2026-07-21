import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd

st.markdown("""
<style>
.stApp {
   
  background: linear-gradient(#134e5e, #71b280, #8AC399);
            
}

</style>
""", unsafe_allow_html=True)


# =================================================
# Augmented IMAGES
# =================================================


st.markdown(
    "<h1 style='color:#ffffff;'>Image Augmentation</h1>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <p style="color:#000000; font-size:18px;">
        Data augmentation techniques expand the training set to improve model generalization and reduce overfitting.
    </p>
    """,
    unsafe_allow_html=True
)


import streamlit as st

st.markdown("""
<style>

.section-card{
    background:#1b1f24;
    border:1px solid #30363d;
    border-radius:15px;
    padding:25px;
    margin-top:20px;
}

.main-title{
    color:#000000;
    font-size:22px;
    font-weight:500;
    margin-bottom:25px;
}

.card-title{
    color:#000000;
    font-size:20px;
    font-weight:500;
    margin-bottom:10px;
}

.card-text{
    color:#000000;
    font-size:18px;
    line-height:1.5;
}

</style>
""", unsafe_allow_html=True)

#st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
Why Augmentation Matters
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="card-title">Expand Dataset</div>

    <div class="card-text">
    Generates multiple variations of each tea leaf image, 
    increasing the amount of training data without collecting new images.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card-title">Reduce Overfitting</div>

    <div class="card-text">
    Prevents the model from memorizing the training images, 
                encouraging it to learn generalizable features such as leaf texture, 
                vein patterns, discoloration, and disease symptoms.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card-title">Increases robustness</div>

    <div class="card-text">
    Simulates real-world conditions where images may be captured from different camera angles, 
    distances, or lighting environments.
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)



# st.markdown(
#     "<h3 style='color:#000000;'>Before vs After Augmentation</h3>",
#     unsafe_allow_html=True
# )
st.image("images/augmented.png",  use_container_width=True)
# =================================================
# DATASET SUMMARY
# =================================================


st.caption("Tea Leaf Quality Detection and Intelligence System")