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

## colors: #c98967 #90A244 #7A7940 f2f2f2 
# background-color: #cccccc;
#  background: linear-gradient(135deg, #134e5e, #71b280);

st.markdown("""
<style>
.stApp {
    
   
    background: linear-gradient(#134e5e, #71b280, #8AC399);
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Tea Leaf Prediction",
    page_icon="🍃",
    layout="wide"
)

st.markdown(
    "<h2 style='color:#ffffff;'>Tea Leaf Prediction</h2>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        color: #000000;
        font-size: 18px;
        font-weight: 500;
        text-align: left;
    ">
        Upload a tea leaf image to classify its quality using the trained deep learning model.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
 #return load_model("cnn_model.keras")

@st.cache_resource
def load_best_model():
    
    return load_model("cnn_resized_model.keras")

model = load_best_model()

# -----------------------------------------------------
# CLASS NAMES
# -----------------------------------------------------

classes = [
    "Highest Quality",
    "Average Quality",
    "Not Good",
    "Diseased"
]

# -----------------------------------------------------
# RECOMMENDATIONS
# -----------------------------------------------------

recommendations = {

    "Highest Quality":
        "Suitable for premium tea production. Good leaf quality.",

    "Average Quality":
        "Suitable for standard commercial tea production.",

    "Not Good":
        "Low-quality leaf. Consider sorting or discarding before processing.",

    "Diseased":
        "Leaf appears diseased. Remove from processing and inspect nearby plants."
}

#----------------------- USAGE --------------

st.markdown(
    "<h3 style='color:#ffffff;'>Usage Instructions</h3>",
    unsafe_allow_html=True
)

col1, col2, col3= st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #4CAF50;
        height:220px;">
        <h2><span style="color:#16a34a;">1️</span></h2>
        <h4>Navigate</h4>
        <p>Open <b>Prediction page</b> from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #4CAF50;
        height:220px;">
        <h2><span style="color:#16a34a;">2️</span></h2>
        <h4>Upload Images</h4>
        <p>Upload one <b>JPG</b>, <b>JPEG</b>, or <b>PNG</b> Tea Leaf images. Click the predict button below</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:2px solid #4CAF50;
        height:220px;">
        <h2><span style="color:#16a34a;">3</span></h2>
        <h4>View Results</h4>
        <p>Review the predicted class with the confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

# #<h2>3️⃣</h2>
# with col4:
#     st.markdown("""
#     <div style="
#         background-color:#fce4ec;
#         padding:20px;
#         border-radius:12px;
#         text-align:center;
#         border:2px solid #E91E63;
#         height:220px;">
#         <h2>4️⃣</h2>
#         <h4>Download</h4>
#         <p>Export the prediction results as a <b>CSV</b> file for record-keeping.</p>
#     </div>
#     """, unsafe_allow_html=True)


# -----------------------------------------------------
# FILE UPLOADER
# -----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Tea Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        image_rgb,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        start = time.time()

        img = cv2.resize(image_rgb, (224,224)) #224 by 224

        img = img.astype("float32") / 255.0

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        end = time.time()

        elapsed = end - start

        predicted_index = np.argmax(prediction)

        predicted_class = classes[predicted_index]

        confidence = prediction[0][predicted_index]

        st.markdown("---")

# -------------- style for below
        st.markdown("""
<style>

/* Metric title */
[data-testid="stMetricLabel"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #71b280 !important;
}

/* Metric value */
[data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: bold !important;
    color: #2E8B57 !important;
}

/* Metric container */
[data-testid="stMetric"] {
    background-color: #F5FCF7;
    border: 2px solid #71b280;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}


/* Increase font size of the metric labels */
[data-testid="stMetricLabel"] p {
    font-size: 16px !important;
    font-weight: bold !important;
    color: #71b280 !important;
}
</style>
""", unsafe_allow_html=True)
        
        #-----------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Class",
            predicted_class
        )

        col2.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        col3.metric(
            "Processing Time",
            f"{elapsed:.3f} sec"
        )

        st.markdown("---")

        st.subheader("Recommendation")

        # st.success(
        #     recommendations[predicted_class]
        # )

        st.markdown(
    f"""
    <div style="
        background-color: #ffffff;
        color: #000000;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #dcfce7;
        font-size: 18px;
        font-weight: 500;
    ">
        {recommendations[predicted_class]}
    </div>
    """,
    unsafe_allow_html=True
)




        st.markdown("---")

        st.subheader("Confidence Scores")

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(
            classes,
            probabilities
        )

        ax.set_ylim(0,1)

        ax.set_ylabel("Probability")

        ax.set_title("Prediction Confidence")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        st.markdown("---")

        # st.subheader("Prediction Summary")

        # st.write(f"""
        # **Predicted Class:** {predicted_class}

        # **Confidence:** {confidence*100:.2f}%

        # **Processing Time:** {elapsed:.3f} seconds

        # **Recommendation:** {recommendations[predicted_class]}
        # """)

        st.subheader("Prediction Summary")

        st.markdown(f"""
        <div style="
        background-color:#ffffff;
        padding:20px;
        border-radius:12px;
        border-left:6px solid #71b280;
        font-size:20px;
        color:#2E8B57;
        line-height:2;
                ">

        <b style="color:#000000;">Predicted Class:</b> {predicted_class}<br>

        <b style="color:#000000;">Confidence:</b> {confidence*100:.2f}%<br>

        <b style="color:#000000;">Processing Time:</b> {elapsed:.3f} seconds<br>

        <b style="color:#000000;">Recommendation:</b> {recommendations[predicted_class]}

        </div>
        """, unsafe_allow_html=True)