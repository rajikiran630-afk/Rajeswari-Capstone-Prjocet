# ============================================================
# MODULE 2 - TITANIC ANALYTICS
# TASK 7 - STRATIFIED TRAIN / TEST SPLIT
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib

# Prevent Tkinter / main-loop errors when running from PowerShell
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n")
print("=" * 60)
print("TASK 7 - STRATIFIED TRAIN / TEST SPLIT")
print("=" * 60)


# ------------------------------------------------------------
# 7.1 Load the cleaned dataset
# ------------------------------------------------------------

print("\nLoading cleaned Titanic dataset...")

df = pd.read_csv("titanic_clean.csv")

print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 7.2 Define target and features
# ------------------------------------------------------------

target = "survived"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget variable:")
print(target)

print("\nFeatures:")
print(X.columns.tolist())


# ------------------------------------------------------------
# 7.3 Check original class balance
# ------------------------------------------------------------

print("\nOriginal Class Distribution")
print("-" * 30)

class_counts = y.value_counts().sort_index()

print(class_counts)

print("\nOriginal Class Percentage")
print("-" * 30)

class_percentages = (
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print(class_percentages)


# ------------------------------------------------------------
# 7.4 Stratified train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 7.5 Display split sizes
# ------------------------------------------------------------

print("\nTrain/Test Split")
print("-" * 30)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")


# ------------------------------------------------------------
# 7.6 Check class balance after stratification
# ------------------------------------------------------------

print("\nTraining Class Distribution")
print("-" * 30)

print(
    y_train.value_counts()
    .sort_index()
)

print("\nTraining Class Percentage")
print("-" * 30)

print(
    y_train.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\nTesting Class Distribution")
print("-" * 30)

print(
    y_test.value_counts()
    .sort_index()
)

print("\nTesting Class Percentage")
print("-" * 30)

print(
    y_test.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 7.7 Interpretation
# ------------------------------------------------------------

print("\nWhy Stratification Matters")
print("-" * 30)

print(
    "The Titanic target variable contains two classes: "
    "survived = 0 and survived = 1."
)

print(
    "The classes are not perfectly balanced, so stratification "
    "ensures that the training and testing sets maintain "
    "approximately the same survival/non-survival proportions "
    "as the original dataset."
)

print(
    "This makes model evaluation more reliable and prevents "
    "the test set from accidentally containing a very different "
    "class distribution from the training data."
)


print("\n" + "=" * 60)
print("TASK 7 - STRATIFIED TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)

# ============================================================
# TASK 8 - PREPROCESSING PIPELINE
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


print("\n")
print("=" * 60)
print("TASK 8 - PREPROCESSING PIPELINE")
print("=" * 60)


# ------------------------------------------------------------
# 8.1 Select modeling features
# ------------------------------------------------------------

model_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[model_features]
y = df["survived"]


print("\nFeatures used for modeling:")
print(model_features)


# ------------------------------------------------------------
# 8.2 Stratified train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain/Test Split")
print("-" * 30)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")


# ------------------------------------------------------------
# 8.3 Define numeric and categorical columns
# ------------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# ------------------------------------------------------------
# 8.4 Numeric preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# 8.5 Categorical preprocessing
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# 8.6 Combine preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 8.7 Fit preprocessing ONLY on training data
# ------------------------------------------------------------

print("\nFitting preprocessing on training data only...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("Training preprocessing completed.")
print("Testing data transformed without refitting.")


# ------------------------------------------------------------
# 8.8 Check processed shapes
# ------------------------------------------------------------

print("\nProcessed Data Shapes")
print("-" * 30)

print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape: "
    f"{X_test_processed.shape}"
)


# ------------------------------------------------------------
# 8.9 Get transformed feature names
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print("\nTransformed Feature Names")
print("-" * 30)

for feature in feature_names:
    print(feature)


# ------------------------------------------------------------
# 8.10 Interpretation
# ------------------------------------------------------------

print("\nPreprocessing Interpretation")
print("-" * 30)

print(
    "Numeric features were processed using median imputation "
    "followed by StandardScaler."
)

print(
    "Categorical features were processed using most-frequent "
    "imputation followed by one-hot encoding."
)

print(
    "The preprocessing steps were fitted only on the training "
    "data and then applied to the test data using transform(). "
    "This prevents test-set information from leaking into training."
)

print(
    "The preprocessing strategy is implemented using a "
    "ColumnTransformer and separate Pipelines for numeric "
    "and categorical features."
)


print("\n" + "=" * 60)
print("TASK 8 - PREPROCESSING COMPLETED")
print("=" * 60)

# ============================================================
# TASK 9 - TRAIN THREE CLASSIFICATION MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


print("\n")
print("=" * 60)
print("TASK 9 - TRAINING THREE CLASSIFIERS")
print("=" * 60)


# ------------------------------------------------------------
# 9.1 Logistic Regression
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")
print("-" * 30)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

print("Logistic Regression trained successfully.")


# ------------------------------------------------------------
# 9.2 Decision Tree
# ------------------------------------------------------------

print("\nTraining Decision Tree...")
print("-" * 30)

decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree_model.fit(
    X_train_processed,
    y_train
)

print("Decision Tree trained successfully.")


# ------------------------------------------------------------
# 9.3 Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")
print("-" * 30)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

print("Random Forest trained successfully.")


# ------------------------------------------------------------
# 9.4 Generate predictions
# ------------------------------------------------------------

logistic_predictions = logistic_model.predict(
    X_test_processed
)

decision_tree_predictions = decision_tree_model.predict(
    X_test_processed
)

random_forest_predictions = random_forest_model.predict(
    X_test_processed
)


# ------------------------------------------------------------
# 9.5 Decision Tree visualization
# ------------------------------------------------------------

print("\nCreating Decision Tree visualization...")

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Titanic Decision Tree")
plt.tight_layout()

plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision Tree saved as decision_tree.png")


# ------------------------------------------------------------
# 9.6 Model summary
# ------------------------------------------------------------

print("\nModel Summary")
print("-" * 30)

print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")


print("\n" + "=" * 60)
print("TASK 9 - THREE CLASSIFIERS TRAINED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# TASK 10 - MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)


print("\n")
print("=" * 60)
print("TASK 10 - MODEL EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# 10.1 Store models and predictions
# ------------------------------------------------------------

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

predictions = {
    "Logistic Regression": logistic_predictions,
    "Decision Tree": decision_tree_predictions,
    "Random Forest": random_forest_predictions
}


# ------------------------------------------------------------
# 10.2 Evaluate each model
# ------------------------------------------------------------

results = []

roc_data = {}


for model_name, model in models.items():

    print("\n" + "-" * 50)
    print(model_name)
    print("-" * 50)

    y_pred = predictions[model_name]

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Probability predictions for ROC/AUC
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test_processed
    )[:, 1]

    auc_score = roc_auc_score(
        y_test,
        y_probability
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score
    }

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc_score:.4f}")

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc_score
    })

    # --------------------------------------------------------
    # Save confusion matrix chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    ).plot(
        ax=ax,
        cmap="Blues"
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved as {filename}"
    )


# ------------------------------------------------------------
# 10.3 Comparison table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# 10.4 ROC Curve
# ------------------------------------------------------------

print("\nCreating ROC curve...")

plt.figure(figsize=(9, 7))

for model_name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{model_name} (AUC = {data['auc']:.3f})"
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Titanic Classification Models")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved as roc_curves.png")


# ------------------------------------------------------------
# 10.5 Save model comparison
# ------------------------------------------------------------

results_df.to_csv(
    "classification_model_comparison.csv",
    index=False
)

print(
    "Model comparison saved as "
    "classification_model_comparison.csv"
)


print("\n" + "=" * 60)
print("TASK 10 - MODEL EVALUATION COMPLETED")
print("=" * 60)

# ============================================================
# TASK 11 - IMBALANCE HANDLING COMPARISON
# ============================================================

from imblearn.over_sampling import SMOTE


print("\n")
print("=" * 60)
print("TASK 11 - IMBALANCE HANDLING COMPARISON")
print("=" * 60)


# ------------------------------------------------------------
# 11.1 Show original class balance
# ------------------------------------------------------------

print("\nOriginal Training Class Balance")
print("-" * 30)

print(
    y_train.value_counts()
    .sort_index()
)

print("\nOriginal Training Class Percentage")
print("-" * 30)

print(
    y_train.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 11.2 Baseline Logistic Regression
# ------------------------------------------------------------

print("\nTraining Baseline Logistic Regression...")
print("-" * 30)

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(
    X_train_processed,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test_processed
)

baseline_precision = precision_score(
    y_test,
    baseline_pred,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_pred,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 11.3 Class Weight Balanced
# ------------------------------------------------------------

print("\nTraining Balanced Logistic Regression...")
print("-" * 30)

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

balanced_model.fit(
    X_train_processed,
    y_train
)

balanced_pred = balanced_model.predict(
    X_test_processed
)

balanced_precision = precision_score(
    y_test,
    balanced_pred,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_pred,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 11.4 SMOTE
# ------------------------------------------------------------

print("\nApplying SMOTE to training data only...")
print("-" * 30)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)


print("Class balance after SMOTE:")

print(
    y_train_smote.value_counts()
    .sort_index()
)


print("\nTraining SMOTE Logistic Regression...")

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_pred = smote_model.predict(
    X_test_processed
)

smote_precision = precision_score(
    y_test,
    smote_pred,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_pred,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 11.5 Comparison table
# ------------------------------------------------------------

imbalance_results = pd.DataFrame({

    "Strategy": [
        "Baseline",
        "Class Weight Balanced",
        "SMOTE"
    ],

    "Precision": [
        baseline_precision,
        balanced_precision,
        smote_precision
    ],

    "Recall": [
        baseline_recall,
        balanced_recall,
        smote_recall
    ],

    "F1": [
        baseline_f1,
        balanced_f1,
        smote_f1
    ]

})


print("\n")
print("=" * 60)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 60)

print(
    imbalance_results.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 11.6 Find best strategy by F1
# ------------------------------------------------------------

best_strategy = imbalance_results.loc[
    imbalance_results["F1"].idxmax()
]

print("\nBest Imbalance Strategy")
print("-" * 30)

print(
    f"Strategy: {best_strategy['Strategy']}"
)

print(
    f"Precision: {best_strategy['Precision']:.4f}"
)

print(
    f"Recall: {best_strategy['Recall']:.4f}"
)

print(
    f"F1: {best_strategy['F1']:.4f}"
)


# ------------------------------------------------------------
# 11.7 Interpretation
# ------------------------------------------------------------

print("\nImbalance Handling Interpretation")
print("-" * 30)

print(
    "The baseline model was trained without any imbalance "
    "handling and provides the reference performance."
)

print(
    "The class_weight='balanced' approach increases the "
    "importance of the minority class during model training "
    "without creating synthetic observations."
)

print(
    "SMOTE creates synthetic minority-class observations, "
    "but it was applied only to the training data. The original "
    "test data was kept completely untouched."
)

print(
    "The strategy with the highest F1 score provides the best "
    "balance between precision and recall for this comparison."
)


# ------------------------------------------------------------
# 11.8 Save comparison
# ------------------------------------------------------------

imbalance_results.to_csv(
    "imbalance_comparison.csv",
    index=False
)

print(
    "\nImbalance comparison saved as "
    "imbalance_comparison.csv"
)


print("\n" + "=" * 60)
print("TASK 11 - IMBALANCE HANDLING COMPLETED")
print("=" * 60)

# ============================================================
# TASK 12 - HYPERPARAMETER TUNING
# ============================================================

print("\n")
print("=" * 60)
print("TASK 12 - HYPERPARAMETER TUNING")
print("=" * 60)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print("\nPreparing Random Forest for GridSearchCV...")
print("-" * 30)

# ------------------------------------------------------------
# 12.1 Random Forest Pipeline
# ------------------------------------------------------------

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                n_jobs=-1
            )
        )
    ]
)

# ------------------------------------------------------------
# 12.2 Parameter Grid
# ------------------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__max_features": ["sqrt", "log2"]
}

print("\nParameter Grid")
print("-" * 30)

for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")

# ------------------------------------------------------------
# 12.3 GridSearchCV
# ------------------------------------------------------------

print("\nRunning GridSearchCV...")
print("-" * 30)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("GridSearchCV completed successfully.")

# ------------------------------------------------------------
# 12.4 Best Parameters
# ------------------------------------------------------------

print("\nBest Parameters")
print("-" * 30)

print(grid_search.best_params_)

# ------------------------------------------------------------
# 12.5 Best Cross-Validation Score
# ------------------------------------------------------------

print("\nBest Cross-Validation F1 Score")
print("-" * 30)

print(f"{grid_search.best_score_:.4f}")

# ------------------------------------------------------------
# 12.6 Best Pipeline
# ------------------------------------------------------------

best_rf_pipeline = grid_search.best_estimator_

# ------------------------------------------------------------
# 12.7 OOB Score
# ------------------------------------------------------------

best_rf_classifier = (
    best_rf_pipeline
    .named_steps["classifier"]
)

print("\nOOB Score")
print("-" * 30)

print(f"{best_rf_classifier.oob_score_:.4f}")

# ------------------------------------------------------------
# 12.8 Test Performance of Tuned Random Forest
# ------------------------------------------------------------

print("\nTuned Random Forest Test Performance")
print("-" * 30)

tuned_rf_predictions = best_rf_pipeline.predict(X_test)

tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_probability = best_rf_pipeline.predict_proba(
    X_test
)[:, 1]

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_probability
)

print(f"Accuracy : {tuned_rf_accuracy:.4f}")
print(f"Precision: {tuned_rf_precision:.4f}")
print(f"Recall   : {tuned_rf_recall:.4f}")
print(f"F1 Score : {tuned_rf_f1:.4f}")
print(f"AUC      : {tuned_rf_auc:.4f}")

# ------------------------------------------------------------
# 12.9 Save Grid Search Results
# ------------------------------------------------------------

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results.to_csv(
    "random_forest_grid_search_results.csv",
    index=False
)

print(
    "\nGridSearch results saved as "
    "random_forest_grid_search_results.csv"
)

print("\n")
print("=" * 60)
print("TASK 12 - HYPERPARAMETER TUNING COMPLETED")
print("=" * 60)

# ============================================================
# TASK 13 - REGRESSION SIDE-TASK
# ============================================================

print("\n")
print("=" * 60)
print("TASK 13 - REGRESSION SIDE-TASK")
print("=" * 60)

# ------------------------------------------------------------
# 13.1 Load the same cleaned Titanic dataset
# ------------------------------------------------------------

print("\nLoading cleaned Titanic dataset...")
print("-" * 30)

regression_df = pd.read_csv("titanic_clean.csv")

print(f"Dataset shape: {regression_df.shape}")

# ------------------------------------------------------------
# 13.2 Define target and features
# ------------------------------------------------------------

# Target variable:
# We want to predict fare.

regression_target = "fare"

# Use the other available features.
# Exclude fare itself and redundant target-related columns.

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = regression_df[regression_features].copy()
y_reg = regression_df[regression_target].copy()

print("\nRegression Target")
print("-" * 30)

print(f"Target: {regression_target}")

print("\nRegression Features")
print("-" * 30)

print(regression_features)

# ------------------------------------------------------------
# 13.3 Train/Test Split
# ------------------------------------------------------------

print("\nRegression Train/Test Split")
print("-" * 30)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print(f"Training rows: {X_reg_train.shape[0]}")
print(f"Testing rows : {X_reg_test.shape[0]}")

# ------------------------------------------------------------
# 13.4 Define numeric and categorical columns
# ------------------------------------------------------------

regression_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]

# ------------------------------------------------------------
# 13.5 Regression Preprocessing
# ------------------------------------------------------------

print("\nRegression Preprocessing")
print("-" * 30)

print(
    "Numeric features: median imputation + StandardScaler"
)

print(
    "Categorical features: most-frequent imputation + OneHotEncoder"
)

regression_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

regression_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            regression_numeric_pipeline,
            regression_numeric_features
        ),
        (
            "categorical",
            regression_categorical_pipeline,
            regression_categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 13.6 Linear Regression Pipeline
# ------------------------------------------------------------

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

# ------------------------------------------------------------
# 13.7 Train Linear Regression
# ------------------------------------------------------------

print("\nTraining Multivariate Linear Regression...")
print("-" * 30)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

print("Linear Regression trained successfully.")

# ------------------------------------------------------------
# 13.8 Make Predictions
# ------------------------------------------------------------

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

# ------------------------------------------------------------
# 13.9 Calculate MAE
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

# ------------------------------------------------------------
# 13.10 Calculate RMSE
# ------------------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

# ------------------------------------------------------------
# 13.11 Calculate R-squared
# ------------------------------------------------------------

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# ------------------------------------------------------------
# 13.12 Calculate Adjusted R-squared
# ------------------------------------------------------------

# Number of observations in the test set
n = len(y_reg_test)

# Number of predictors after preprocessing
#
# Get the transformed feature matrix to determine
# the actual number of regression predictors.

X_reg_test_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
)

p = X_reg_test_transformed.shape[1]

adjusted_r2 = (
    1
    - (
        (1 - r2) * (n - 1)
        / (n - p - 1)
    )
)

# ------------------------------------------------------------
# 13.13 Display Regression Metrics
# ------------------------------------------------------------

print("\nRegression Model Evaluation")
print("-" * 30)

print(f"MAE         : {mae:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adjusted_r2:.4f}")

print(f"\nNumber of test observations (n): {n}")
print(f"Number of predictors (p): {p}")

# ------------------------------------------------------------
# 13.14 Create Residuals
# ------------------------------------------------------------

residuals = y_reg_test - y_reg_pred

# ------------------------------------------------------------
# 13.15 Residual Plot
# ------------------------------------------------------------

print("\nCreating residual plot...")
print("-" * 30)

plt.figure(figsize=(9, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Linear Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    "fare_regression_residual_plot.png",
    dpi=300
)

plt.close()

print(
    "Residual plot saved as "
    "fare_regression_residual_plot.png"
)

# ------------------------------------------------------------
# 13.16 Basic Heteroscedasticity Check
# ------------------------------------------------------------

print("\nHeteroscedasticity Check")
print("-" * 30)

# Divide predictions into low, medium and high predicted fare
# groups and compare the residual spread.

residual_check = pd.DataFrame(
    {
        "predicted_fare": y_reg_pred,
        "residual": residuals
    }
)

residual_check["prediction_group"] = pd.qcut(
    residual_check["predicted_fare"],
    q=3,
    duplicates="drop"
)

residual_std_by_group = (
    residual_check
    .groupby(
        "prediction_group",
        observed=False
    )["residual"]
    .std()
)

print("\nResidual standard deviation by predicted-fare group:")
print(residual_std_by_group.round(4))

# ------------------------------------------------------------
# 13.17 Written Interpretation
# ------------------------------------------------------------

print("\nRegression Interpretation")
print("-" * 30)

print(
    f"The multivariate linear regression model achieved an "
    f"MAE of {mae:.4f}, RMSE of {rmse:.4f}, R² of {r2:.4f}, "
    f"and Adjusted R² of {adjusted_r2:.4f}."
)

print(
    "The residual plot is used to check whether the residuals "
    "are randomly distributed around zero."
)

# Compare the smallest and largest residual standard deviation.
residual_std_values = residual_std_by_group.dropna().values

if len(residual_std_values) >= 2:

    min_std = residual_std_values.min()
    max_std = residual_std_values.max()

    ratio = max_std / min_std if min_std != 0 else np.inf

    if ratio > 2:
        print(
            "The residual spread changes substantially across "
            "predicted fare levels, suggesting evidence of "
            "heteroscedasticity."
        )
    else:
        print(
            "The residual spread is reasonably similar across "
            "predicted fare levels, so there is no strong evidence "
            "of heteroscedasticity from this exploratory check."
        )

else:

    print(
        "There were not enough prediction groups to perform "
        "a reliable residual-spread comparison."
    )

# ------------------------------------------------------------
# 13.18 Save Regression Metrics
# ------------------------------------------------------------

regression_metrics = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "Adjusted_R2"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            adjusted_r2
        ]
    }
)

regression_metrics.to_csv(
    "regression_metrics.csv",
    index=False
)

print(
    "\nRegression metrics saved as "
    "regression_metrics.csv"
)

print("\n")
print("=" * 60)
print("TASK 13 - REGRESSION SIDE-TASK COMPLETED")
print("=" * 60)

# ============================================================
# TASK 14 - FINAL MODEL COMPARISON AND RECOMMENDATION
# ============================================================

print("\n")
print("=" * 60)
print("TASK 14 - FINAL MODEL COMPARISON")
print("=" * 60)

# ------------------------------------------------------------
# 14.1 Classification Results
# ------------------------------------------------------------

# These are the results obtained in Task 10.

classification_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        0.8090,
        0.7640,
        0.8202
    ],

    "Precision": [
        0.7833,
        0.7600,
        0.7812
    ],

    "Recall": [
        0.6912,
        0.5588,
        0.7353
    ],

    "F1": [
        0.7344,
        0.6441,
        0.7576
    ],

    "AUC": [
        0.8610,
        0.8374,
        0.8179
    ]
})

print("\nClassification Model Results")
print("-" * 60)

print(
    classification_results.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# 14.2 Regression Results
# ------------------------------------------------------------

# Use the actual Task 13 results.

regression_results = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}

print("\nRegression Model Results")
print("-" * 60)

print(
    f"MAE         : {mae:.4f}"
)

print(
    f"RMSE        : {rmse:.4f}"
)

print(
    f"R²          : {r2:.4f}"
)

print(
    f"Adjusted R² : {adjusted_r2:.4f}"
)

# ------------------------------------------------------------
# 14.3 Final Comparison Table
# ------------------------------------------------------------

# Classification and regression metrics are intentionally
# kept as separate metric groups.

final_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Linear Regression"
    ],

    # Classification metric group
    "Classification_Accuracy": [
        0.8090,
        0.7640,
        0.8202,
        np.nan
    ],

    "Classification_Precision": [
        0.7833,
        0.7600,
        0.7812,
        np.nan
    ],

    "Classification_Recall": [
        0.6912,
        0.5588,
        0.7353,
        np.nan
    ],

    "Classification_F1": [
        0.7344,
        0.6441,
        0.7576,
        np.nan
    ],

    "Classification_AUC": [
        0.8610,
        0.8374,
        0.8179,
        np.nan
    ],

    # Regression metric group
    "Regression_MAE": [
        np.nan,
        np.nan,
        np.nan,
        mae
    ],

    "Regression_RMSE": [
        np.nan,
        np.nan,
        np.nan,
        rmse
    ],

    "Regression_R2": [
        np.nan,
        np.nan,
        np.nan,
        r2
    ],

    "Regression_Adjusted_R2": [
        np.nan,
        np.nan,
        np.nan,
        adjusted_r2
    ]
})

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 60)

print(
    final_comparison.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# 14.4 Save Final Comparison Table
# ------------------------------------------------------------

final_comparison.to_csv(
    "final_model_comparison.csv",
    index=False
)

print(
    "\nFinal model comparison saved as "
    "final_model_comparison.csv"
)

# ------------------------------------------------------------
# 14.5 Final Recommendation
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL MODEL RECOMMENDATION")
print("=" * 60)

recommendation = (
    "Among the three classification models, Random Forest is "
    "recommended for deployment because it achieved the highest "
    "test accuracy of 0.8202 and the highest F1 score of 0.7576. "
    "It also achieved a recall of 0.7353, meaning it identified "
    "a larger proportion of surviving passengers than Logistic "
    "Regression and Decision Tree. Logistic Regression achieved "
    "the highest AUC of 0.8610, showing strong overall ranking "
    "performance, but Random Forest provided the best balance "
    "of accuracy, recall, and F1 on the held-out test set. "
    "Therefore, Random Forest is selected as the final classifier "
    "for deployment."
)

print("\n" + recommendation)

# Save recommendation to text file

with open(
    "final_model_recommendation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FINAL MODEL RECOMMENDATION\n"
        "==========================\n\n"
    )

    file.write(
        recommendation
    )

print(
    "\nFinal recommendation saved as "
    "final_model_recommendation.txt"
)

print("\n")
print("=" * 60)
print("TASK 14 - FINAL MODEL COMPARISON COMPLETED")
print("=" * 60)


# ============================================================
# TASK 15 - SAVE COMPLETE BEST PIPELINE
# ============================================================

print("\n")
print("=" * 60)
print("TASK 15 - SAVE COMPLETE BEST PIPELINE")
print("=" * 60)

# ------------------------------------------------------------
# 15.1 Select Best Classifier
# ------------------------------------------------------------

print("\nSelected deployment model")
print("-" * 30)

print("Model: Random Forest")
print("Reason: Highest test accuracy and F1 score")

# ------------------------------------------------------------
# 15.2 Prepare Raw Modeling Data
# ------------------------------------------------------------

# IMPORTANT:
# We use RAW feature columns here.
#
# The preprocessing will happen INSIDE the Pipeline.
#
# Therefore, the saved pipeline can receive raw,
# unprocessed data later.

model_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target_column = "survived"

deployment_df = pd.read_csv(
    "titanic_clean.csv"
)

X_deployment = deployment_df[
    model_features
].copy()

y_deployment = deployment_df[
    target_column
].copy()

print("\nRaw deployment features")
print("-" * 30)

print(model_features)

# ------------------------------------------------------------
# 15.3 Stratified Split
# ------------------------------------------------------------

X_deploy_train, X_deploy_test, y_deploy_train, y_deploy_test = (
    train_test_split(
        X_deployment,
        y_deployment,
        test_size=0.20,
        random_state=42,
        stratify=y_deployment
    )
)

print("\nDeployment Train/Test Split")
print("-" * 30)

print(
    f"Training rows: {len(X_deploy_train)}"
)

print(
    f"Testing rows : {len(X_deploy_test)}"
)

# ------------------------------------------------------------
# 15.4 Create Fresh Preprocessing Pipeline
# ------------------------------------------------------------

deployment_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

deployment_categorical_features = [
    "sex",
    "embarked"
]

deployment_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

deployment_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

deployment_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            deployment_numeric_pipeline,
            deployment_numeric_features
        ),
        (
            "categorical",
            deployment_categorical_pipeline,
            deployment_categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 15.5 Create Complete Pipeline
# ------------------------------------------------------------

# IMPORTANT:
# This is NOT just the Random Forest.
#
# The complete pipeline contains:
#
# Raw Data
#    ↓
# Imputation
#    ↓
# Encoding
#    ↓
# Scaling
#    ↓
# Random Forest
#
# Therefore the saved object can accept raw data.

full_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            deployment_preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# ------------------------------------------------------------
# 15.6 Fit Complete Pipeline
# ------------------------------------------------------------

print("\nTraining complete deployment pipeline...")
print("-" * 30)

full_pipeline.fit(
    X_deploy_train,
    y_deploy_train
)

print(
    "Complete pipeline fitted successfully."
)

# ------------------------------------------------------------
# 15.7 Test Complete Pipeline
# ------------------------------------------------------------

deployment_predictions = full_pipeline.predict(
    X_deploy_test
)

deployment_accuracy = accuracy_score(
    y_deploy_test,
    deployment_predictions
)

print("\nComplete Pipeline Test Accuracy")
print("-" * 30)

print(
    f"Accuracy: {deployment_accuracy:.4f}"
)

# ------------------------------------------------------------
# 15.8 Save Complete Pipeline
# ------------------------------------------------------------

import joblib

pipeline_filename = (
    "best_titanic_classification_pipeline.joblib"
)

joblib.dump(
    full_pipeline,
    pipeline_filename
)

print("\nPipeline Saved")
print("-" * 30)

print(
    f"File: {pipeline_filename}"
)

# ------------------------------------------------------------
# 15.9 Confirm File Exists
# ------------------------------------------------------------

import os

if os.path.exists(
    pipeline_filename
):

    file_size = os.path.getsize(
        pipeline_filename
    )

    print(
        f"Pipeline file exists successfully."
    )

    print(
        f"File size: {file_size:,} bytes"
    )

else:

    print(
        "ERROR: Pipeline file was not created."
    )

# ------------------------------------------------------------
# 15.10 Reload Saved Pipeline
# ------------------------------------------------------------

print("\nReloading saved pipeline...")
print("-" * 30)

loaded_pipeline = joblib.load(
    pipeline_filename
)

print(
    "Pipeline reloaded successfully."
)

# ------------------------------------------------------------
# 15.11 Predict Using RAW Test Data
# ------------------------------------------------------------

# IMPORTANT:
# We deliberately provide RAW X_deploy_test.
#
# We do NOT manually:
# - impute
# - scale
# - encode
#
# The loaded pipeline handles all of those steps.

reloaded_predictions = loaded_pipeline.predict(
    X_deploy_test
)

reloaded_accuracy = accuracy_score(
    y_deploy_test,
    reloaded_predictions
)

print("\nReloaded Pipeline Verification")
print("-" * 30)

print(
    f"Predictions generated: "
    f"{len(reloaded_predictions)}"
)

print(
    f"Accuracy after reload: "
    f"{reloaded_accuracy:.4f}"
)

# ------------------------------------------------------------
# 15.12 Verify Predictions Are Identical
# ------------------------------------------------------------

predictions_match = np.array_equal(
    deployment_predictions,
    reloaded_predictions
)

print(
    f"Predictions identical before/after reload: "
    f"{predictions_match}"
)

if predictions_match:

    print(
        "\nSUCCESS: Saved pipeline works correctly "
        "on raw input after reloading."
    )

else:

    print(
        "\nWARNING: Predictions changed after reloading."
    )

# ------------------------------------------------------------
# 15.13 Create a Raw New Passenger Example
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("RAW NEW DATA PREDICTION TEST")
print("=" * 60)

# This example contains NO preprocessing.
#
# The pipeline itself handles:
# - missing values
# - categorical encoding
# - scaling

new_passenger = pd.DataFrame({
    "pclass": [1],
    "sex": ["female"],
    "age": [30],
    "sibsp": [0],
    "parch": [0],
    "fare": [80],
    "embarked": ["C"]
})

print("\nRaw new passenger:")
print(new_passenger)

new_prediction = loaded_pipeline.predict(
    new_passenger
)

print("\nPrediction")
print("-" * 30)

if new_prediction[0] == 1:

    print(
        "Predicted survival: SURVIVED (1)"
    )

else:

    print(
        "Predicted survival: DID NOT SURVIVE (0)"
    )

# Probability prediction

if hasattr(
    loaded_pipeline,
    "predict_proba"
):

    new_probability = (
        loaded_pipeline
        .predict_proba(
            new_passenger
        )[0][1]
    )

    print(
        f"Predicted survival probability: "
        f"{new_probability:.4f}"
    )

# ------------------------------------------------------------
# 15.14 Create Verification Script
# ------------------------------------------------------------

verification_script = r'''
import pandas as pd
import joblib

# Load saved complete pipeline
pipeline = joblib.load(
    "best_titanic_classification_pipeline.joblib"
)

# Raw new passenger data
new_passenger = pd.DataFrame({
    "pclass": [1],
    "sex": ["female"],
    "age": [30],
    "sibsp": [0],
    "parch": [0],
    "fare": [80],
    "embarked": ["C"]
})

# Predict directly from raw data
prediction = pipeline.predict(
    new_passenger
)

print("Prediction:", prediction)

if hasattr(pipeline, "predict_proba"):
    probability = pipeline.predict_proba(
        new_passenger
    )[0][1]

    print(
        "Survival probability:",
        round(probability, 4)
    )
'''

with open(
    "verify_saved_pipeline.py",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        verification_script
    )

print(
    "\nVerification script saved as "
    "verify_saved_pipeline.py"
)

# ------------------------------------------------------------
# 15.15 Final Pipeline Summary
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("SAVED PIPELINE SUMMARY")
print("=" * 60)

print(
    "\nPipeline components:"
)

print(
    "1. Median imputation for numeric features"
)

print(
    "2. StandardScaler for numeric features"
)

print(
    "3. Most-frequent imputation for categorical features"
)

print(
    "4. One-hot encoding for categorical features"
)

print(
    "5. Random Forest classifier"
)

print(
    "\nThe preprocessing steps and classifier "
    "are saved together as one Pipeline object."
)

print(
    "The saved pipeline can accept raw, "
    "unprocessed input data."
)

print("\n")
print("=" * 60)
print("TASK 15 - COMPLETE PIPELINE SAVED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# FINAL ANALYTICS MODULE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ANALYTICS MODULE - TASKS 1 TO 15 COMPLETED")
print("=" * 70)

print("\nImportant output files:")

print("1. titanic.csv")
print("2. titanic_clean.csv")
print("3. age_histogram.png")
print("4. age_boxplot.png")
print("5. fare_histogram.png")
print("6. fare_boxplot.png")
print("7. correlation_heatmap.png")
print("8. multivariate_survival_heatmap.png")
print("9. standardization_before_after.png")
print("10. decision_tree.png")
print("11. logistic_regression_confusion_matrix.png")
print("12. decision_tree_confusion_matrix.png")
print("13. random_forest_confusion_matrix.png")
print("14. roc_curves.png")
print("15. fare_regression_residual_plot.png")
print("16. classification_model_comparison.csv")
print("17. imbalance_comparison.csv")
print("18. random_forest_grid_search_results.csv")
print("19. regression_metrics.csv")
print("20. final_model_comparison.csv")
print("21. final_model_recommendation.txt")
print("22. best_titanic_classification_pipeline.joblib")
print("23. verify_saved_pipeline.py")

print("\n")
print("=" * 70)
print("ALL ANALYTICS TASKS COMPLETED")
print("=" * 70)