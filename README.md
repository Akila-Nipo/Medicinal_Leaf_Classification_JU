# 🌿 Medicinal Leaf Classification - ML Project

A Machine Learning project for classifying medicinal plant leaves using HOG feature extraction and classical ML classifiers like **SGDClassifier** and **SVC**, built with Python and `scikit-learn`.

## 📂 Project Overview

This project focuses on classifying images of medicinal leaves into categories such as **Aloevera**, **Bamboo**, **Coriander**, **Ginger**, etc., using classical image processing and machine learning techniques.

## 🧠 Workflow Summary

1. 📁 **Image Dataset Preparation**: Load and preprocess images from `MedicinalLeafDataset`
2. 📐 **Feature Extraction**: Use HOG (Histogram of Oriented Gradients) to extract features
3. 🧪 **Model Training**: Train models using `SGDClassifier` and `SVC`
4. 🧾 **Evaluation**: Assess with confusion matrix and classification report
5. 💾 **Saving Models**: Store models using `joblib` for reuse
6. 🔍 **Prediction**: Apply the trained model to unseen images

## 🏷️ Dataset Format

Your dataset should follow this directory structure:

```plaintext
MedicinalLeafDataset/
├── AloeveraLeaf/
│   ├── image1.jpg
│   └── ...
├── BambooLeaf/
├── CorienderLeaf/
├── GingerLeaf/
├── MangoLeaf/
├── PepperLeaf/
└── TurmericLeaf/
```
## 🔧 How to Use

### 1. Update Data Path in Code

In `MedicinalLeafClassifier.py`, update the dataset path:

```python
data_path = r'F:\Path\To\Your\MedicinalLeafDataset'
```
Replace with the path where your MedicinalLeafDataset folder is located.

## 2. Run the Code

```bash
python MedicinalLeafClassifier.py
```
The script will:

Resize images

Extract HOG features

Train classifiers (SGD/SVM)

Evaluate and save the model

## 🔍 Feature Extraction with HOG

```python
from skimage.feature import hog

features = hog(image, 
               orientations=9, 
               pixels_per_cell=(8, 8), 
               cells_per_block=(2, 2), 
               visualize=True)
```
## 🤖 Model Training

```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
clf.fit(X_train, y_train)
```

## 📊 Evaluation

Accuracy, precision, recall, and F1-score using `classification_report()`  
Confusion matrix for visual evaluation:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
```
## 💾 Saving and Loading Models

```python
import joblib

# Save model
joblib.dump(clf, 'model.pkl')

# Load model
clf = joblib.load('model.pkl')
```
## 📦 Dependencies

Install required packages:

```bash
pip install scikit-learn scikit-image joblib matplotlib numpy
```
## 👥 Authors

- **Akila Nipo** – Roll: 368  
- **Rubayed Al Islam** – Roll: 370

---

## 📚 Course Info

- **Course**: Machine Learning Laboratory (CSE-458)  
- **University**: Jahangirnagar University  
- **Instructor**: Dr. Liton Jude Rozario *(Professor)*

  ## 🧠 Key Insights

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx.png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(1).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(2).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(3).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(4).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(5).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(6).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(7).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(8).png" alt="project logo">
</p>
    <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(8).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(9).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(10).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(11).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(12).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(13).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(14).png" alt="project logo">
</p>

  <p align="center">
    <img src="https://github.com/Akila-Nipo/django_temporary/blob/main/ML_Project_Presentation_Slide_368%20-%20Copy.pptx%20(15).png" alt="project logo">
</p>
