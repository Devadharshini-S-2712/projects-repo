Machine Learning Models Used
Logistic Regression
Random Forest Classifier

Project Workflow
Data Loading
Load the CSV file and explore shape, columns, and distribution.

Preprocessing

1.Drop unnecessary columns like id and Time (if present)

2.Normalize the Amount feature using StandardScaler

3.Balance the dataset using undersampling

Model Training
Split the data into training and test sets (80:20)
Train Logistic Regression and Random Forest models
Evaluate performance using metrics

Evaluation Metrics

Accuracy
1.Precision, Recall, F1-score
2.Confusion Matrix
3.ROC Curve with AUC Score

Visualizations
1.Countplot of class distribution
2.Confusion Matrix Heatmap
3.ROC Curve plot

