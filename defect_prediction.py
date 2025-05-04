import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, precision_score, f1_score
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('dataset.csv')

print(data.head())

print(f"Missing values:\n{data.isnull().sum()}")

numeric_cols = data.select_dtypes(include=[np.number]).columns
text_cols = data.select_dtypes(include=[object]).columns

# Handle missing values for numeric columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Handle text columns 
tfidf = TfidfVectorizer(stop_words='english', max_features=500)  
text_data = tfidf.fit_transform(data['report'])  

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Convert the sparse matrix to a DataFrame
text_data_df = pd.DataFrame(text_data.toarray(), columns=tfidf.get_feature_names_out())

# Combine the processed text data with the rest of the dataset
data_imputed = pd.concat([data.drop(columns=text_cols), text_data_df], axis=1)

# Check label distribution to see if it's imbalanced
label_columns = [col for col in data_imputed.columns if col.startswith('type_')]  
label_counts = data_imputed[label_columns].sum()
print(f"Label distribution:\n{label_counts}")

# Plot the label distribution
label_counts.plot(kind='bar', title='Label Distribution')
plt.show()

# Check if any labels have only one class
for col in label_columns:
    unique_classes = data_imputed[col].value_counts()
    print(f"Unique classes in label {col}:\n{unique_classes}\n")
    
    if len(unique_classes) == 1:
        print(f"Warning: Label {col} has only one class: {unique_classes.index[0]}")

# Remove labels with only one class
valid_labels = [col for col in label_columns if data_imputed[col].nunique() > 1]
print(f"Valid labels (those with more than one class): {valid_labels}")

# Features and target (after removing labels with only one class)
X = data_imputed.drop(columns=label_columns)
y = data_imputed[valid_labels]  # Multi-label target with valid labels

# Ensure X and y have the same number of rows (this should already be the case)
assert X.shape[0] == y.shape[0], "Mismatch between feature matrix (X) and target matrix (y) dimensions."

# Feature scaling: Standardize the feature matrix
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split again to create validation and test sets (50% of temp for validation and 50% for test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hyperparameter tuning grid for Logistic Regression, SVM, Perceptron, and DNN
param_dist_log_reg = {
    'estimator__C': uniform(0.01, 100),  # Random values for C
    'estimator__solver': ['lbfgs', 'saga']
}

param_grid_svm = {
    'estimator__C': [0.01, 0.1, 1, 10, 100],
    'estimator__gamma': ['scale', 'auto'],
    'estimator__kernel': ['rbf', 'linear']
}

param_dist_perceptron = {
    'estimator__alpha': uniform(0.0001, 0.1),  # Random values for alpha
    'estimator__max_iter': [500, 1000, 1500]
}

param_dist_dnn = {
    'estimator__hidden_layer_sizes': [(100,), (200,), (256, 128)],
    'estimator__learning_rate_init': uniform(0.001, 0.1),  # Random values for learning rate
    'estimator__max_iter': [200, 500, 1000]
}

# Define models for multi-label classification
log_reg_model = MultiOutputClassifier(LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga'))
svm_model = MultiOutputClassifier(SVC(probability=True, class_weight='balanced'))
svm_model.fit(X_scaled, y)
perceptron_model = MultiOutputClassifier(Perceptron(max_iter=5000))
dnn_model = MultiOutputClassifier(MLPClassifier(max_iter=2000, learning_rate_init=0.001, early_stopping=True))

# Train models only if there are at least two classes for each label
if all(y_train[col].sum() > 0 for col in valid_labels):
    # RandomizedSearchCV for Logistic Regression
    random_search_log_reg = RandomizedSearchCV(estimator=log_reg_model, param_distributions=param_dist_log_reg, n_iter=10, cv=5, n_jobs=-1, verbose=2)
    random_search_log_reg.fit(X_train, y_train)
    print(f"Best parameters for Logistic Regression: {random_search_log_reg.best_params_}")

    # RandomizedSearchCV for SVM
    grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=2)
    grid_search_svm.fit(X_scaled, y)
    print(f"Best parameters for SVM: {grid_search_svm.best_params_}")

    # RandomizedSearchCV for Perceptron
    random_search_perceptron = RandomizedSearchCV(estimator=perceptron_model, param_distributions=param_dist_perceptron, n_iter=10, cv=5, n_jobs=-1, verbose=2)
    random_search_perceptron.fit(X_train, y_train)
    print(f"Best parameters for Perceptron: {random_search_perceptron.best_params_}")

    # RandomizedSearchCV for DNN
    random_search_dnn = RandomizedSearchCV(estimator=dnn_model, param_distributions=param_dist_dnn, n_iter=10, cv=5, n_jobs=-1, verbose=2)
    random_search_dnn.fit(X_train, y_train)
    print(f"Best parameters for DNN: {random_search_dnn.best_params_}")

    # Evaluate models using multi-label metrics
    def evaluate_tuned(model, X_val, y_val):
        y_pred = model.predict(X_val)

        # Hamming Loss
        hamming = hamming_loss(y_val, y_pred)

        # Precision@k
        precision_at_k = precision_score(y_val, y_pred, average='micro')

        # Micro-F1 and Macro-F1
        micro_f1 = f1_score(y_val, y_pred, average='micro')
        macro_f1 = f1_score(y_val, y_pred, average='macro')

        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Micro-F1: {micro_f1:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Precision@k: {precision_at_k:.4f}")

    # Evaluate all tuned models
    print("\nEvaluating Logistic Regression (Tuned)...")
    evaluate_tuned(random_search_log_reg.best_estimator_, X_val, y_val)

    print("\nEvaluating SVM (Tuned)...")
    evaluate_tuned(grid_search_svm.best_estimator_, X_val, y_val)

    print("\nEvaluating Perceptron (Tuned)...")
    evaluate_tuned(random_search_perceptron.best_estimator_, X_val, y_val)

    print("\nEvaluating DNN (Tuned)...")
    evaluate_tuned(random_search_dnn.best_estimator_, X_val, y_val)

    # Save the trained models
    joblib.dump(random_search_log_reg.best_estimator_, './defect_models/log_reg_model_defect.pkl')
    joblib.dump(grid_search_svm.best_estimator_, './defect_models/svm_model_defect.pkl')
    joblib.dump(random_search_perceptron.best_estimator_, './defect_models/perceptron_model_defect.pkl')
    joblib.dump(random_search_dnn.best_estimator_, './defect_models/dnn_model_defect.h5')

else:
    print("Training aborted due to insufficient classes in the training data.")