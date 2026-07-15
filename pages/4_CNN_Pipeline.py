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

st.markdown("""
<style>

.pipeline-title{
    font-size:40px;
    font-weight:bold;
    color:#ffffff;
    text-align:left;
    margin-bottom:30px;
}

.pipeline-card{
    background:#1b1f24;
    border:1px solid #30363d;
    border-radius:15px;
    padding:20px;
    margin:12px auto;
    width:80%;
    text-align:center;
    transition:0.3s;
}

.pipeline-card:hover{
    border-color:#2dd4bf;
    transform:translateY(-3px);
    box-shadow:0px 8px 20px rgba(45,212,191,.2);
}

.card-title{
    color:white;
    font-size:28px;
    font-weight:bold;
    margin-bottom:8px;
}

.card-desc{
    color:#9ca3af;
    font-size:18px;
}

.arrow{
    text-align:center;
    color:#ffffff;
    font-size:34px;
    line-height:1.2;
}

.output-card{
    background:#0f172a;
    border:2px solid #2dd4bf;
    border-radius:15px;
    padding:20px;
    width:80%;
    margin:auto;
}

.output-title{
    color:#2dd4bf;
    font-size:26px;
    font-weight:bold;
    text-align:center;
    margin-bottom:15px;
}

.output-class{
    color:white;
    font-size:20px;
    margin:8px 0;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="pipeline-title">CNN Classification Pipeline</div>', unsafe_allow_html=True)

steps = [
("1. Tea Leaf Image","Input image uploaded by the user."),
("2. Resize (224 × 224)","images resized to 224×224×3 (RGB) - dimensions expected by the CNN."),
("3. Normalize","Pixel values scaled from 0–255 to 0–1."),
("4. Convolution Block 1","Extracts low-level features such as edges, colors and veins."),
("5. ReLU Activation","Removes negative values and introduces non-linearity."),
("6. Max Pooling","Reduces feature map size while retaining important information."),
("7. Convolution Block 2","Learns textures, shapes and leaf structure."),
("8. ReLU Activation","Keeps important activations."),
("9. Max Pooling","Further reduces dimensionality."),
("10. Convolution Block 3","Detects disease spots, discoloration and damaged edges."),
("11. Global Average Pooling","Converts feature maps into a compact feature vector."),
("12. Dense Layers","Combines learned features for classification."),
("13. Softmax Layer","Calculates probabilities for each tea leaf class.")
]

for title, desc in steps:

    st.markdown(f"""
    <div class="pipeline-card">
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
    </div>

    <div class="arrow">
        ↓
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="output-card">

<div class="output-title">
Prediction Output
</div>

<div class="output-class">🟢 High Quality</div>

<div class="output-class">🟡 Average Quality</div>

<div class="output-class">🟠 Not Good for Tea</div>

<div class="output-class">🔴 Diseased</div>

</div>
""", unsafe_allow_html=True)