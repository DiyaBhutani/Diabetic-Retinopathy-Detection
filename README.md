# Diabetic Retinopathy Detection using Machine Learning

## Overview
Diabetic Retinopathy (DR) is a diabetes-related eye disease and one of the leading causes of preventable blindness. Early detection plays a crucial role in preventing vision loss.

This project implements a classical machine learning–based system to detect diabetic retinopathy from retinal fundus images. The system focuses on image preprocessing, handcrafted feature extraction, data balancing, model training, and evaluation using medically relevant metrics.

---

## Objectives
- Enhance low-contrast retinal images for improved feature visibility  
- Extract meaningful handcrafted features from medical images  
- Handle class imbalance commonly found in medical datasets  
- Train and compare multiple machine learning classifiers  
- Evaluate model performance using healthcare-oriented metrics  

---

## Dataset
- Retinal fundus images with corresponding diagnosis labels  
- Labels represent the severity level of diabetic retinopathy  
- Dataset information is provided in `messidor_data.csv`  

---

## Methodology

### 1. Image Preprocessing
- Images are resized to a fixed resolution of 128×128 pixels  
- The green channel is extracted, as it best highlights retinal blood vessels and lesions  
- Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to enhance local contrast and reveal fine retinal details  

---

### 2. Feature Extraction
Each image is converted into a numerical feature vector using handcrafted features:

- Texture features using Gray Level Co-occurrence Matrix (GLCM):
  - Contrast
  - Energy
  - Homogeneity
  - Correlation

- Intensity-based features:
  - Normalized grayscale histogram (16 bins)

- Statistical features:
  - Mean pixel intensity
  - Standard deviation of pixel intensity

These features capture both structural and intensity variations caused by diabetic retinopathy.

---

### 3. Data Balancing
- The dataset is balanced using SMOTE (Synthetic Minority Oversampling Technique)
- This prevents model bias toward majority classes and improves sensitivity for minority classes

---

### 4. Model Training
Two supervised machine learning models are trained and compared:

- Random Forest Classifier  
  - Hyperparameters optimized using GridSearchCV  
  - Provides strong performance and robustness  

- Gaussian Naïve Bayes  
  - Used as a baseline probabilistic classifier  

---

### 5. Evaluation Metrics
Model performance is evaluated using:

- Accuracy  
- Sensitivity (Recall)  
- Specificity  
- Confusion Matrix  
- Classification Report  

These metrics are particularly important for medical diagnosis tasks.

---

## Visualization
The project includes visual analysis using:

- Histogram of feature distributions  
- Bar chart comparing model accuracies  
- Line plots showing feature trends  
- Scatter plots showing feature relationships  
- Confusion matrix heatmaps for each classifier  

---

## CLAHE Demonstration
A separate module demonstrates and compares:

- Histogram Equalization  
- Contrast Limited Adaptive Histogram Equalization (CLAHE)  

This highlights why CLAHE is better suited for medical images with uneven illumination.

---

## Results
- Random Forest classifier outperformed Naïve Bayes in overall accuracy and sensitivity  
- CLAHE significantly improved contrast and feature visibility  
- SMOTE improved detection of minority classes  

---

## Future Enhancements
- Extend the project using deep learning models such as CNNs  
- Deploy the trained model as a web-based screening application  
- Add severity-level prediction and probability-based risk scoring  

---

## Technologies Used
- Python  
- OpenCV  
- scikit-image  
- scikit-learn  
- imbalanced-learn  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

---

## Conclusion
This project demonstrates a complete machine learning pipeline for diabetic retinopathy detection using classical techniques. It emphasizes interpretability, proper preprocessing, and reliable evaluation, making it suitable for academic study and healthcare-oriented applications.
