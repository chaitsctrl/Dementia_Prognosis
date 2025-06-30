# Dementia_Prognosis

This repository contains multiple Dementia AI-based modules, including **Facial Emotion Recognition (FER)**, **Large Language Model (LLM) integration**, and **MRI-based dementia analysis**. Each module is structured to run independently or as part of the complete system.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Folder Structure](#folder-structure)  
3. [Installation](#installation)  
4. [Setup Instructions](#setup-instructions)  
5. [Running the Modules](#running-the-modules)  
6. [Usage](#usage)  
7. [Troubleshooting](#troubleshooting)  
8. [Dependencies](#dependencies)  

---

## **Project Overview**

The project consists of three main modules:

1. **FER**: Classifies facial emotions from uploaded images using a pre-trained model.
2. **LLM**: Analyzes dementia-related data and interacts with it using a language model.
3. **MRI**: Predicts the severity of dementia using MRI scans and machine learning models.
4. **FAISS Index**: A placeholder for FAISS-based vector search utilities.

The project uses **Streamlit** for the user interface and integrates machine learning and deep learning models.

---

## **Folder Structure**

```plaintext
Dementia-Analysis/
│
├── faiss_index/                 # FAISS index-related files
│
├── FER/                         # Facial Emotion Recognition
│   ├── images/                  # Input images for testing
│   ├── models/                  # Model files for FER
│   ├── transforms/              # Image preprocessing utilities
│   ├── emotion.t7               # Pre-trained model weights
│   └── app.py                   # Streamlit app for FER
│
├── LLM/                         # Language Model for dementia analysis
│   ├── Dementia_Data.json       # JSON dataset
│   ├── EDI_dementia.ipynb       # Notebook for data analysis
│   ├── LLM_data_dementia.csv    # CSV file for dementia data
│   ├── yaml-editor-online.yaml  # YAML configurations
│   └── app.py                   # Streamlit app for LLM
│
├── MRI/                         # MRI-based dementia severity detection
│   ├── main.py                  # MRI analysis script
│   ├── svm_model_vgg16.joblib   # Pre-trained SVM model
│   ├── vgg16_weights.h5         # Pre-trained VGG16 weights
│   └── app.py                   # Streamlit app for MRI
│
├── requirements.txt             # Python dependencies
└── app.py                       # Combined Streamlit app
```

---

## **Installation**

### **1. Prerequisites**
- **Python 3.8+**
- **Git**  
- Recommended: Virtual environment (venv).

---

### **2. Steps to Install**

1. Clone this repository:
   ```bash
   git clone <repository-url>
   or else download .zip file and extract the content
   cd Dementia-Analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Mac/Linux
   venv\Scripts\activate          # On Windows
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Setup Instructions**

- Verify the required model files in their respective directories:
  - `FER/emotion.t7` for the FER module.
  - `MRI/vgg16_weights.h5` and `MRI/svm_model_vgg16.joblib` for MRI analysis.
- Ensure the datasets (`LLM/Dementia_Data.json` and `LLM/LLM_data_dementia.csv`) are valid and available.

---

## **Running the Modules**

You can run either the **combined app** or the individual modules.

### **1. Run the Combined App**

Execute the root-level `app.py` to run the integrated Streamlit interface:

```bash
streamlit run app.py
```

---

### **2. Run Individual Modules**

- **Facial Emotion Recognition (FER)**:
   ```bash
   cd FER
   streamlit run app.py
   ```

- **LLM for Dementia Data**:
   ```bash
   cd LLM
   streamlit run app.py
   or else run the EDI_dementia.ipynb file
   ```

- **MRI-based Dementia Detection**:
   ```bash
   cd MRI
   streamlit run main.py
   ```

---

## **Usage**

### **FER Module**:
1. Upload an image containing a face.
2. The system will predict and display the detected **emotion**.

### **LLM Module**:
1. Query dementia-related data.
2. The app will fetch insights from the uploaded CSV file and generate relative response.

### **MRI Module**:
1. Upload an MRI scan image.
2. The system will classify the **severity of dementia** as:
   - Non-Demented
   - Very Mild Demented
   - Mild Demented
   - Moderate Demented  

---

## **Troubleshooting**

1. **Virtual Environment Issues**:
   - Ensure the environment is activated:
     ```bash
     source venv/bin/activate
     ```

2. **Missing Dependencies**:
   - Reinstall the requirements:
     ```bash
     pip install -r requirements.txt
     ```

3. **FER Model Errors**:
   - Verify that `FER/emotion.t7` exists in the correct directory.

4. **LLM Data Issues**:
   - Ensure the JSON and CSV files are correctly formatted.

5. **Streamlit Installation**:
   - If Streamlit is missing:
     ```bash
     pip install streamlit
     ```

---

## **Dependencies**

The following packages are required for this project:

- **streamlit**
- **torch**
- **torchvision**
- **scikit-image**
- **numpy**
- **matplotlib**
- **joblib**
- **pandas**
- **langchain**
- **faiss-cpu**
- **Pillow**

Refer to the `requirements.txt` file for the complete list.
