# Breast-Cancer

## Task 4: Logistic Regression Classifier for Breast Cancer Prediction

***

### 1. Project Objective and Overview

* **Objective**: To build and evaluate a **binary classification** model using **Logistic Regression** to predict breast cancer diagnosis.
* **Target Variable**: $\mathbf{y}$ (Diagnosis): Malignant (0) or Benign (1).
* **Dataset**: **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains 569 samples and 30 numerical features describing cell nuclei characteristics (e.g., mean radius, texture, perimeter).

***

### 2. Methodology and Implementation

The project was executed using Python with **Scikit-learn**, **Pandas**, and **Matplotlib**, following these key steps:

1.  **Data Loading & Splitting**:
    * The dataset was loaded and split into features ($\mathbf{X}$) and target ($\mathbf{y}$).
    * The data was then divided into a **Training Set** (80%) and a **Testing Set** (20%) using `train_test_split` to ensure the model's performance could be accurately measured on unseen data.
2.  **Feature Standardization**:
    * Features were scaled using $\mathbf{StandardScaler}$. This step is crucial for Logistic Regression as it standardizes the input features to have a mean of 0 and a standard deviation of 1, preventing features with large values from dominating the model.
3.  **Model Training**:
    * A **Logistic Regression** model was initialized and trained (`fit`) on the scaled training data.
    * The model calculates the optimal weights for the features by minimizing the log-loss (cross-entropy).

***

### 3. Core Algorithm: The Sigmoid Function

The foundation of the Logistic Regression classifier is the **Sigmoid function** (or Logistic function).

* **Function**: It takes the raw, linear output ($z$) of the model and maps it to a value $P$ between 0 and 1, which represents the **probability** of the positive class (Benign).
* **Formula**:
    $$P = \frac{1}{1 + e^{-z}}$$
* **Classification**: The model predicts the positive class if $P \ge \text{Threshold}$ (default $0.5$) and the negative class otherwise.

***

### 4. Model Evaluation and Tuning

#### A. Key Metrics (Default Threshold: 0.5)

| Metric | Calculation | Typical Value | Interpretation |
| :--- | :--- | :---: | :--- |
| **ROC-AUC** | Area under the ROC Curve | $\approx 0.99$ | Excellent separation ability between Malignant and Benign classes across all thresholds. |
| **Accuracy** | Overall Correct Predictions | $\approx 97\%$ | High overall proportion of correct diagnoses. |
| **Precision** | $\frac{TP}{TP+FP}$ | High ($\approx 97\%$) | When the model predicts 'Malignant', it is correct $\approx 97\%$ of the time. |
| **Recall (Sensitivity)** | $\frac{TP}{TP+FN}$ | High ($\approx 98\%$) | The model correctly identifies $\approx 98\%$ of all actual 'Benign' cases. |

#### B. Confusion Matrix

The matrix shows the counts of correct and incorrect predictions:

$$
\begin{pmatrix}
\text{True Negatives (TN)} & \text{False Positives (FP)} \\
\text{False Negatives (FN)} & \text{True Positives (TP)}
\end{pmatrix}
$$


#### C. Threshold Tuning

* **Process**: The default classification threshold of $0.5$ was **tuned** by analyzing the **ROC Curve**.
* **Purpose**: In medical diagnosis, minimizing **False Negatives (FN)** (missing a malignant case) is often critical. Tuning the threshold allows balancing **Recall** (to reduce FN) and **Precision** based on the specific application's risk tolerance.
* **Result**: An **optimal threshold** (e.g., $0.40$) was identified to potentially maximize a balance metric like the Geometric Mean of Recall and Specificity.

***

### 5. Conclusion

The Logistic Regression model demonstrated **high performance** (ROC-AUC $\approx 0.99$) in classifying breast masses, indicating its strong capability as a diagnostic aid. The project successfully demonstrated all phases of a binary classification task, from data preparation and model training to detailed performance evaluation and threshold optimization.
