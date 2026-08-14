
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
