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
# ------------------------------------
# PAGE CONFIGURATION
# ------------------------------------
st.set_page_config(
    page_title="Dataset",
    page_icon="🍃",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    "<h1 style='color:#ffffff;'>Data Analysis & Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")


st.markdown("""
<h3 style='text-align:left; color:#ffffff;'>
Global Tea Industry Overview
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
font-size:18px;
color:#000000;
text-align:left;
">
Tea is one of the world's most valuable agricultural commodities,
supporting millions of farmers and contributing significantly to the global economy.
</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Production Value",
    "US$17B+"
)

col2.metric(
    "Global Tea Trade",
    "US$9.5B"
)

col3.metric(
    "World Production",
    "7.05M Tonnes"
)

col4.metric(
    "People Employed",
    "13M+"
)

st.divider()



# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

col1, col2 = st.columns(2)

# Production Share
with col1:

    st.subheader("Global Tea Production Share")

    production = pd.DataFrame({
        "Country": [
            "China",
            "India",
            "Kenya",
            "Others"
        ],
        "Share": [
            53,
            18,
            8.5,
            20.5
        ]
    })

    fig, ax = plt.subplots(figsize=(6,5))

    ax.bar(
        production["Country"],
        production["Share"]
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0,60)

    for i, value in enumerate(production["Share"]):
        ax.text(i, value+1, f"{value}%", ha='center', fontsize=10)

    st.pyplot(fig)


# Employment
with col2:

    st.subheader("Tea Sector Employment")

    employment = pd.DataFrame({
        "Category":[
            "Smallholder Farmers",
            "Others"
        ],
        "Value":[
            60,
            40
        ]
    })

    fig, ax = plt.subplots(figsize=(6,5))

    ax.pie(
        employment["Value"],
        labels=employment["Category"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Tea Produced by Smallholders")

    st.pyplot(fig)

st.divider()


#--------------------------------
# Dataset
# -----------------------------------------
st.markdown(
    "<h2 style='color:#ffffff;'>Dataset Statistics</h2>",
    unsafe_allow_html=True
)


st.markdown("---")

# -------------------------------------------------
# DATASET PATH
# -------------------------------------------------

dataset_path = "tea_dataset_resized"

classes = sorted(os.listdir(dataset_path))

# -------------------------------------------------
# COUNT IMAGES
# -------------------------------------------------
image_counts = {}
total_images = 0

for cls in classes:

    folder = os.path.join(dataset_path, cls)

    count = len(os.listdir(folder))

    image_counts[cls] = count

    total_images += count

#===============================================
classes = []
counts = []

# Count images in each class
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        classes.append(folder)
        counts.append(len(os.listdir(folder_path)))

# Create colors
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

# # Create figure
# fig, ax = plt.subplots(figsize=(6, 4))

# ax.bar(classes, counts, color=colors)

# ax.set_title("Dataset Class Distribution", fontsize=11)
# ax.set_xlabel("Tea Quality Class", fontsize=10)
# ax.set_ylabel("Number of Images", fontsize=10)

# ax.tick_params(axis='x', labelrotation=45, labelsize=9)
# ax.tick_params(axis='y', labelsize=9)

# # Display in Streamlit
# st.pyplot(fig)


# =================================================
# BAR CHART
# =================================================





# =================================================
# BAR CHART + PIE CHART SIDE BY SIDE
# =================================================

col1, col2 = st.columns(2)

# ------------------------
# BAR CHART
# ------------------------
with col1:

    classes = []
    counts = []

    # Count images in each class
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):
            classes.append(folder)
            counts.append(len(os.listdir(folder_path)))

    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(classes, counts, color=colors)

    ax.set_title("Dataset Class Distribution", fontsize=11)
    ax.set_xlabel("Tea Quality Class", fontsize=10)
    ax.set_ylabel("Number of Images", fontsize=10)

    ax.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    st.pyplot(fig)

# ------------------------
# PIE CHART
# ------------------------
with col2:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.pie(
        image_counts.values(),
        labels=image_counts.keys(),
        autopct='%1.1f%%',
        startangle=90
    )

    ax2.set_title("Dataset Distribution")

    st.pyplot(fig2)

st.markdown("---")


# =================================================
# SAMPLE IMAGES
# =================================================

st.markdown(
    "<h3 style='color:#ffffff;'>Sample Images by Class</h3>",
    unsafe_allow_html=True
)

# st.write(
#     "The images below show one example from each tea leaf quality class."
# )

st.markdown(
    """
    <p style="color:#000000; font-size:20px;">
        The images below show one example from each tea leaf quality class.
    </p>
    """,
    unsafe_allow_html=True
)

#st.markdown("---")
st.markdown("---")

for cls in classes:
    st.subheader(cls)

    folder = os.path.join(dataset_path, cls)
    image_files = os.listdir(folder)[84:88]   # First 4 images

    cols = st.columns(4)

    for i, image_name in enumerate(image_files):
        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cols[i].image(
            image,
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

