# Team 22471A05 — Decoding Dementia: Alzheimer’s Classification Using VGG16 + GWO

---

## Team Info

- 22471A0521 — **G Revanth Sai** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Model implementation, feature extraction using VGG16, training & evaluation._

- 22471A0506 — **Ch Jaswanth Reddy** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Dataset preparation, preprocessing, balancing techniques, documentation._

- 22471A0527 — **K Adithya** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Grey Wolf Optimizer (GWO) feature selection integration and experimentation._

- 22471A0528 — **M Karthik** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Performance analysis, result visualization (confusion matrix, ROC, metrics)._

---

## Abstract

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder affecting memory and cognition. Traditional diagnosis methods often struggle due to high-dimensional MRI data and class imbalance.  
To overcome these issues, this project proposes a hybrid deep learning framework combining **VGG16-based feature extraction** with **Grey Wolf Optimizer (GWO)** for selecting the most relevant diagnostic features.  

A CNN classifier is trained on optimized features for multi-class prediction across four AD stages. Experiments on a Kaggle MRI dataset demonstrate outstanding performance, achieving **99.36% accuracy** and an F1-score of **99%**, surpassing existing models.

---

## Paper Reference (Inspiration)

👉 **[STCNN model integrating SMOTE-Tomek with CNN for Alzheimer’s classification – Anjali et al., 2024](Paper URL here)**  
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper

This project improves upon existing STCNN approaches by:

- Using **VGG16 transfer learning** for deep feature extraction  
- Applying **Grey Wolf Optimization (GWO)** to remove redundant features  
- Improving efficiency, interpretability, and generalization  
- Achieving higher accuracy (**99.36%**) than baseline models such as VGG19 and DenseNet169  

---

## About the Project

This project automatically detects and classifies Alzheimer’s stages from MRI brain scans.

### What it does:
- Takes an MRI scan as input  
- Extracts deep features using VGG16  
- Selects the most important features using GWO  
- Predicts one of four Alzheimer’s stages  

### Why it is useful:
- Helps in early diagnosis  
- Supports clinical decision-making  
- Reduces human diagnostic error  

### Workflow:

**MRI Input → Preprocessing → VGG16 Feature Extraction → GWO Feature Selection → CNN Classifier → Stage Prediction Output**

---

## Dataset Used

👉 **Alzheimer MRI 4-Class Dataset (Kaggle)**  
[Dataset Link](Dataset URL)

### Dataset Details:

- MRI images grouped into 4 classes:
  - Non Demented (NOD)
  - Very Mild Demented (VMD)
  - Mild Demented (MD)
  - Moderate Demented (MOD)

- Balanced subset created with **200 images per class**  
- Total: **800 MRI scans**

---

## Dependencies Used

- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Scikit-learn  
- Matplotlib  
- SMOTE-Tomek (imbalanced-learn)  
- Grey Wolf Optimizer implementation  

---

## EDA & Preprocessing

Preprocessing steps applied:

- Resizing all images to **224×224**
- Pixel normalization using `preprocess_input()`
- One-hot encoding for labels
- Train-test split (80:20)
- Dataset balancing using sampling methods

---

## Model Training Info

- Feature extractor: **VGG16 pretrained CNN**
- Feature selector: **Grey Wolf Optimizer**
- Final classifier: Custom CNN with Dense + Dropout layers
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Training epochs: **30 epochs**

---

## Model Testing / Evaluation

Evaluation metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC-AUC curves  

The model was tested across all four classes with near-perfect discrimination.

---

## Results

Key achievements:

- **Accuracy: 99.36%**
- **F1-score: 99%**
- ROC-AUC Macro Average: **0.9958**
- Moderate Demented recall: **1.00**

Confusion matrix shows very few misclassifications, mainly between Mild and Very Mild Demented stages.

---

## Limitations & Future Work

### Current limitations:

- Dataset size is limited (only Kaggle MRI scans)
- Minor confusion between Mild and Very Mild stages
- Needs validation on real clinical hospital datasets

### Future improvements:

- Add Grad-CAM++ explainability
- Test on multimodal datasets (MRI + PET + clinical records)
- Deploy as real-time diagnostic web application

---

## Deployment Info

Deployment can be done using:

- Flask / FastAPI backend  
- Streamlit frontend for demo  
- Upload MRI scan → Get Alzheimer stage prediction  
- Future scope: Mobile + Hospital integration  

---
