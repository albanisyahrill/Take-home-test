# Take-home-test

Berikut adalah template **README.md** untuk mereproduksi model yang ditrain di notebook:

---

# **Model Reproduction Guide**

This repository contains the code and resources required to reproduce the training and evaluation of the machine learning model described in this project. Follow the steps below to set up your environment and reproduce the results.

---

## **Table of Contents**
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)
4. [Training and Evaluating the Model](#training-and-evaluating-the-model)

---

## **Requirements**
Ensure you have the following dependencies installed:
- Python == 3.10.12
- TensorFlow >= 2.17.1
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Random
- Collections
- Pillow

You can install all dependencies using the `requirements_ipynb.txt` file provided:
```bash
pip install -r requirements_ipynb.txt
```

---

## **Dataset**
1. **Download the dataset**: 
   - You can download the dataset used in this project from [Dataset Link](https://huggingface.co/datasets/garythung/trashnet).
   - Ensure the dataset is stored in the `data/` directory or update the paths in the notebook accordingly.

2. **Structure of the dataset**:
   ```
   data/
   └── train/
   ```

---

## **Setup Instructions**
1. Clone this repository:
   ```bash
   git clone https://github.com/albanisyahrill/Take-home-test.git
   cd your-repo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements_ipynb.txt
   ```
---

## **Training and Evaluating the Model**
1. Open the notebook:
   ```bash
   jupyter notebook Take_Home_Test.ipynb
   ```

2. Follow the steps in the notebook to:
   - Load the dataset.
   - Exploratory Image Analysis.
   - Preprocess the dataset.
   - Define the model architecture.
   - Train the model using the specified hyperparameters.
   - Evaluate the model using metrics Accuracy, Precision, Recall, F1-score
   - Save the trained model.
