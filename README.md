# Detecting-fraudulent-transactions-with-XGBoost-and-Keras

# Goal
The objective of this project is to detect fraudulent transactions in credit card data using machine learning models. Specifically, it compares the performance of an XGBoost classifier and a deep learning model built with Keras, evaluating their abilities to successfully identify rare fraud cases in a highly imbalanced dataset.

# Tools Used
- Data Source: [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
- Data Analysis & Visualization: pandas, numpy, matplotlib, seaborn

Machine Learning Models:
- XGBoost (with class balancing via scale_pos_weight)
- Deep Learning model using TensorFlow Keras (Sequential API, Dense and Dropout layers)
- Metrics & Evaluation: scikit-learn (classification report, accuracy, precision, recall, F1-score, ROC-AUC), matplotlib (ROC curve plotting)

# Conclusions
- Both the XGBoost and Keras models achieved very high accuracy (~99.9%) on the test set.
- XGBoost outperformed the deep learning model in terms of F1-score, recall, and overall ability to handle the class imbalance, with an F1-score of 0.81 vs. 0.76 for Keras.
- Precision and AUC-ROC were also higher for XGBoost (Precision: 0.82, AUC-ROC: 0.90) compared to the Keras model (Precision: 0.75, AUC-ROC: 0.89).
- Feature selection based on feature importance helped improve model performance by reducing the dataset to the most relevant features.
- The project demonstrates that with proper handling of class imbalance, classical gradient boosting models like XGBoost can be highly effective for fraud detection tasks—even compared to deep learning approaches.
