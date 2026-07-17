# Tea Leaf Quality Detection and intelligence system

## Project Overview
To develop an intelligent deep learning-based system that automatically classifies **Tea Leaves** according to quality and health status, providing accurate assessments and recommendations to support tea harvesting and processing decisions through an analytics dashboard, model comparison, and explainable outputs.


## Problem Statements
Tea quality is highly dependent on the maturity and health of harvested leaves. Traditionally, tea leaf grading is performed manually, making the process time-consuming, subjective, and prone to inconsistencies. Incorrect classification of leaf quality can reduce the value of the final product, while the inclusion of diseased leaves can negatively affect tea quality and processing outcomes.

To address these challenges, there is a need for an automated and reliable system that can accurately assess tea leaf quality and identify diseased leaves. This project proposes the development of a Tea Quality Detection and Intelligence System using Convolutional Neural Networks (CNNs) and transfer learning models to classify tea leaves into 4 categories: Highest quality, Good quality, Average Quality, Bad Quality and Diseased. The system will provide quality assessments and recommendations to support more efficient and consistent tea grading and decision-making.

________________________________________


## Full System Overview
- **Quality Recommendation:** would give outputs like, Highest quality, Average quality, Diseased leaves etc.
- **Dashboard:** Presents classification results, confidence scores, recommendations, and model performance
- **Batch Image Processing** (if available)
- **Model Comparison Module**


### Key Components
- **System Objective:** Provide quality recommendations and quality analytics to support decision-making in tea production.
- **Deep Learning:** Multi-class image classification of tea leaf quality using Convolutional Neural Networks (CNNs). Predict the quality of tea leaves from images.
- **Web Interface:** Streamlit or Roboflow **
- **Reporting Module:** Presents results in a clear and understandable format.


___________________________________________


### Technology
**Machine Learning**

- TensorFlow / Keras
- Scikit-learn
- NumPy / Pandas
- OpenCV
- Python

**Web Interface**
- Streamlit


____________________________________________

## Project Architecture
                           ┌─────────────────┐
                           │ Tea Leaf Images │
                           └────────┬────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │ Data Collection & Label │
                      └────────┬────────────────┘
                               │
                               ▼
                    ┌───────────────────────────┐
                    │ Image Preprocessing       │
                    │ • Resize (224×224)        │
                    │ • Rescale (1/255)         │
                    │ • Augmentation            │
                    └──────────┬────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       ┌─────────────────┐       ┌──────────────────┐
       │ Custom CNN      │       │ Transfer Learning│
       │                 │       │ EfficientNet     |
       │                 |       | ImagineNet       |   
       └────────┬────────┘       └────────┬─────────┘
                │                         │
                └─────────┬───────────────┘
                          ▼
              ┌──────────────────────────┐
              │ Model Comparison Module  │
              │ Accuracy / Precision     │
              │ Recall / F1-Score        |
              | Confusion Matrix         |          
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Best Trained Model       │
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Prediction Engine        │
              │ Highest Quality          │
              | Bad Quality              │
              │ Average Quality          │
              │ Diseased                 |
              │                          | 
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Recommendation           │
              │ • Highest Quality        │
              │ • Average Quality        │
              │ • Can't use              |
              │ • Diseased               | 
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Streamlit Dashboard      │
              │ • Upload Image           │
              │ • Prediction Results     │
              │ • Confidence Score       │
              │ • Analytics & Reports    │
              └──────────────────────────┘

__________________________________

## Setup Instructions

- 


____________________________


## Metrics

Model Metrics

| Metric | Precision | Recall | F1-Score | Support |
|:-------|----------:|-------:|---------:|--------:|
| Average_quality | 0.71 | 0.76 | 0.74 | 101 |
| Diseased | 0.97 | 0.97 | 0.97 | 124 |
| Not_good_for_tea | 0.75 | 0.78 | 0.76 | 104 |
| Good_quality | 0.82 | 0.75 | 0.78 | 126 |