# 📌 Customer Churn Prediction using Decision Tree

## Project Overveiw
This project builds a **Machine Learning model"" to predict whether a bank custiomer will **churn (exit)** or not.
A **Decision Tree Classifier** is used, covering the complete ML workflow including **data preprocessing, EDA, model training, evaluation, visualization, and model saving**.

---

## Problem Statement
Customer churn is a critical issue for banks.
The objective of this project is to predict:
- `Exited = 1` → Customer is likely to leave
- `Exited = 0` → Customer will stay


This helps organization take preventive actions to retain customers.


---

## Dataset Information
- Dataset: Bank Customer Churn Dataset
- Each row represents one customer
- Target column: `Exited`

---

## Technologies & Libraries Used
- Python
- Numpy
- Pandas
- Matplotlib
- Plotly
- Scikit-Learn
- Pickle

---

## Project Workflow

### 1️⃣ Data Loading &  Validation
- Dataset loaded using Pandas
- Dataset shape checked
- Dataset information (`info()`) checked
- Missing values checked
    ✔️ No null values found

---

### 2️⃣ Data Preprocessing
- `Gender` column encoded using **Label Encoding**
- `Geography` column:
    - Value counts analyzed
    - Converted using **One-Hot Encoding**
    - Encoded columns added to main dataset
    - Original `Geography` column dropped
- Removed unnecessary columns:
    - `RowNumber`
    - `Surname`

---

## 3️⃣ Exploratory Data Analysis (EDA)
- Target column (`Exited`) visualized using **pie chart**
    - Churned (1): **20.4%**
    - Not Churned (0): **79.6%**
![Pie Plot](./images/Existed%20pie.png)

- Dataset correlation anlyzed
- Statistical summary checked using `describe()`

➡️ The dataset is **imbalanced**
---

### 4️⃣ Feature & Target Separation
- Features → `x`
- Target → `y`
- Checked `x.head()` and `y.head()`

---

### 5️⃣ Train-Test Split
- Dataset split into:
    - **70% Training data**
    - **30% Testing data**


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

---

## Model Training
* Algorithm: **Decision Tree Classifier**
* Model trained using training data

---

## Model Evaluation
* Accuracy
8 Accuracy Score: 0.794

---

## Classification Report
```yml
precision    recall  f1-score   support

0       0.88      0.86      0.87      2416
1       0.48      0.53      0.50       584

accuracy                           0.80      3000
macro avg       0.68      0.69      0.69      3000
weighted avg    0.80      0.80      0.80      3000
```
➡️ The model performs well for **non-churn customers**
➡️ Recall for churned customers can be improved

---

## Confusion Matrix
The confusion matrix was visualized using `imshow()`after converting it into a DataFrame.
```text
| Actual \ Predicted | Negative | Positive |
| ------------------ | -------- | -------- |
| Actual Negative    | 2082     | 334      |
| Actual Positive    | 284      | 300      |
```
![Visualized Confussion Matrix](./images/Visualized%20Confussion%20Matrix.png)

---

## Model Saving
* The trained model was saved using **Pickle (.pkl)** format
```text
churn_decision_tree_model.pkl
```
This allow easy reuse in APIs or deployment environments.

---

## Future Improvements
* Handle class imbalance using **SMOTE** or class_weight
* Perform **hyperparameter tuning**
* Try advanced models:
    - Random Forest
    - XGBoost
* **Implement Artiicial Neural Network (ANN)** for better performance
* Deploy the model using Flask/FastAPI
* Create a web-based dashboard for predictions

---

## 🧑‍💻 Author
**Sourov Roy**
Aspiring Machine Learning / AI Engineer

---


# How to Use
```bash
git clone <https://github.com/sourov-roy-ovi/customer-churn-prediction.git>
pip install -r requirements.txt
```

---