import pandas as pd
import seaborn as sns


# ============================================================
# MODULE 2 - ANALYTICS PIPELINE
# TASK 1 - LOAD AND PROFILE TITANIC DATASET
# ============================================================

print("=" * 60)
print("MODULE 2 - TITANIC ANALYTICS")
print("=" * 60)


# ============================================================
# 1. LOAD TITANIC DATASET
# ============================================================

print("\nLoading Titanic dataset...")

df = sns.load_dataset("titanic")

print("Titanic dataset loaded successfully!")


# ============================================================
# 2. SAVE OFFLINE FALLBACK
# ============================================================

df.to_csv("titanic.csv", index=False)

print("Offline fallback saved as titanic.csv")


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\nDataset Shape")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Shape:", df.shape)


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\nDataset Information")
print("=" * 60)

df.info()


# ============================================================
# 5. BASIC STATISTICS
# ============================================================

print("\nBasic Statistics")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\nMissing Values")
print("=" * 60)

missing_count = df.isnull().sum()

missing_percentage = (df.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

print(missing_report[missing_report["Missing Count"] > 0])


# ============================================================
# # ============================================================
# 7. MISSING VALUE HANDLING
# ============================================================

print("\nMissing Value Handling")
print("=" * 60)


# ------------------------------------------------------------
# AGE
# Missing percentage = 19.87%
# Rule: 5% - 30% -> Impute
# Strategy: Median imputation
# ------------------------------------------------------------

age_missing_pct = df["age"].isnull().mean() * 100

print(f"age missing percentage: {age_missing_pct:.2f}%")
print("Decision: Median imputation because missingness is between 5% and 30%.")

age_median = df["age"].median()

df["age"] = df["age"].fillna(age_median)


# ------------------------------------------------------------
# EMBARKED
# Missing percentage = 0.22%
# Rule: Below 5% -> Drop rows
# ------------------------------------------------------------

embarked_missing_pct = df["embarked"].isnull().mean() * 100

print(f"\nembarked missing percentage: {embarked_missing_pct:.2f}%")
print("Decision: Drop rows because missingness is below 5%.")


# ------------------------------------------------------------
# EMBARK_TOWN
# Missing percentage = 0.22%
# Rule: Below 5% -> Drop rows
# ------------------------------------------------------------

embark_town_missing_pct = df["embark_town"].isnull().mean() * 100

print(f"embark_town missing percentage: {embark_town_missing_pct:.2f}%")
print("Decision: Drop rows because missingness is below 5%.")


# Drop rows with missing embarked or embark_town
df = df.dropna(subset=["embarked", "embark_town"])


# ------------------------------------------------------------
# DECK
# Missing percentage = 77.22%
# Rule: Above 30% -> Drop column
# ------------------------------------------------------------

deck_missing_pct = df["deck"].isnull().mean() * 100

print(f"\ndeck missing percentage: {deck_missing_pct:.2f}%")
print("Decision: Drop the deck column because more than 30% is missing.")

df = df.drop(columns=["deck"])


# ============================================================
# 8. VERIFY CLEAN DATA
# ============================================================

print("\nAfter Cleaning")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nRemaining Missing Values")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 9. SAVE CLEANED DATA
# ============================================================

df.to_csv("titanic_clean.csv", index=False)

print("\nCleaned Titanic dataset saved successfully!")
print("File: titanic_clean.csv")


# ============================================================
# TASK 2 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("TASK 2 - CLEANING COMPLETED")
print("=" * 60)

# ============================================================
# TASK 3 - UNIVARIATE ANALYSIS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n============================================================")
print("TASK 3 - UNIVARIATE ANALYSIS")
print("============================================================")


# ============================================================
# 1. AGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(df["age"], bins=20, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("age_histogram.png")
plt.close()

print("\nAge histogram saved as age_histogram.png")


# ============================================================
# 2. AGE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["age"])

plt.title("Age Box Plot")
plt.xlabel("Age")

plt.tight_layout()
plt.savefig("age_boxplot.png")
plt.close()

print("Age box plot saved as age_boxplot.png")


# ============================================================
# 3. FARE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(df["fare"], bins=30, kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("fare_histogram.png")
plt.close()

print("Fare histogram saved as fare_histogram.png")


# ============================================================
# 4. FARE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["fare"])

plt.title("Fare Box Plot")
plt.xlabel("Fare")

plt.tight_layout()
plt.savefig("fare_boxplot.png")
plt.close()

print("Fare box plot saved as fare_boxplot.png")


# ============================================================
# 5. IQR OUTLIERS - AGE
# ============================================================

Q1_age = df["age"].quantile(0.25)
Q3_age = df["age"].quantile(0.75)

IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = df[
    (df["age"] < lower_age) |
    (df["age"] > upper_age)
]

print("\nAge IQR Analysis")
print("------------------------------")
print("Q1:", round(Q1_age, 2))
print("Q3:", round(Q3_age, 2))
print("IQR:", round(IQR_age, 2))
print("Lower Bound:", round(lower_age, 2))
print("Upper Bound:", round(upper_age, 2))
print("Number of Age Outliers:", len(age_outliers))


# ============================================================
# 6. IQR OUTLIERS - FARE
# ============================================================

Q1_fare = df["fare"].quantile(0.25)
Q3_fare = df["fare"].quantile(0.75)

IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = df[
    (df["fare"] < lower_fare) |
    (df["fare"] > upper_fare)
]

print("\nFare IQR Analysis")
print("------------------------------")
print("Q1:", round(Q1_fare, 2))
print("Q3:", round(Q3_fare, 2))
print("IQR:", round(IQR_fare, 2))
print("Lower Bound:", round(lower_fare, 2))
print("Upper Bound:", round(upper_fare, 2))
print("Number of Fare Outliers:", len(fare_outliers))


# ============================================================
# 7. FARE MEAN, MEDIAN AND MODE
# ============================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\nFare Statistics")
print("------------------------------")
print("Mean:", round(fare_mean, 2))
print("Median:", round(fare_median, 2))
print("Mode:", round(fare_mode, 2))


# ============================================================
# 8. FARE SKEWNESS INTERPRETATION
# ============================================================

print("\nFare Distribution Interpretation")
print("------------------------------")

if fare_mean > fare_median > fare_mode:
    print(
        "Fare is right-skewed because "
        "Mean > Median > Mode."
    )

elif fare_mean < fare_median < fare_mode:
    print(
        "Fare is left-skewed because "
        "Mean < Median < Mode."
    )

else:
    print(
        "Fare distribution does not follow a simple "
        "mean-median-mode ordering."
    )


# ============================================================
# TASK 3 COMPLETED
# ============================================================

print("\n============================================================")
print("TASK 3 - UNIVARIATE ANALYSIS COMPLETED")
print("============================================================")

# ============================================================
# TASK 4 - BIVARIATE ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("TASK 4 - BIVARIATE ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 4.1 Survival Rate by Sex
# ------------------------------------------------------------

print("\nSurvival Rate by Sex")
print("-" * 30)

survival_by_sex = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_sex)

# ------------------------------------------------------------
# 4.2 Survival Rate by Passenger Class
# ------------------------------------------------------------

print("\nSurvival Rate by Passenger Class")
print("-" * 30)

survival_by_pclass = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_pclass)

# ------------------------------------------------------------
# 4.3 Survival Rate by Sex and Passenger Class
# ------------------------------------------------------------

print("\nSurvival Rate by Sex and Passenger Class")
print("-" * 30)

survival_by_sex_pclass = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_sex_pclass)

# ------------------------------------------------------------
# 4.4 Boolean Masking
# ------------------------------------------------------------

print("\nBoolean Masking Examples")
print("-" * 30)

female_first_class = df[
    (df["sex"] == "female") &
    (df["pclass"] == 1)
]

female_first_class_survival = (
    female_first_class["survived"].mean() * 100
)

print(
    f"Female passengers in 1st class survival rate: "
    f"{female_first_class_survival:.2f}%"
)

male_third_class = df[
    (df["sex"] == "male") &
    (df["pclass"] == 3)
]

male_third_class_survival = (
    male_third_class["survived"].mean() * 100
)

print(
    f"Male passengers in 3rd class survival rate: "
    f"{male_third_class_survival:.2f}%"
)

# ------------------------------------------------------------
# 4.5 Correlation Matrix
# ------------------------------------------------------------
import numpy as np
print("\nCorrelation Matrix")
print("-" * 30)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix.round(3))

# ------------------------------------------------------------
# 4.6 Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()

print("\nCorrelation heatmap saved as correlation_heatmap.png")

# ------------------------------------------------------------
# 4.7 Find Two Strongest Correlations
# ------------------------------------------------------------

print("\nTwo Strongest Correlations")
print("-" * 30)

corr_pairs = correlation_matrix.copy()

# Remove self-correlations
for column in corr_pairs.columns:
    corr_pairs.loc[column, column] = np.nan

# Convert matrix into pairs
corr_pairs_long = (
    corr_pairs
    .stack()
    .reset_index()
)

corr_pairs_long.columns = [
    "feature_1",
    "feature_2",
    "correlation"
]

# Remove duplicate pairs
corr_pairs_long["pair"] = corr_pairs_long.apply(
    lambda row: tuple(
        sorted([row["feature_1"], row["feature_2"]])
    ),
    axis=1
)

corr_pairs_long = corr_pairs_long.drop_duplicates("pair")

# Rank by absolute correlation
corr_pairs_long["absolute_correlation"] = (
    corr_pairs_long["correlation"].abs()
)

top_two = (
    corr_pairs_long
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
    .head(2)
)

for index, row in top_two.iterrows():
    print(
        f"{row['feature_1']} <-> {row['feature_2']}: "
        f"{row['correlation']:.3f}"
    )

# ------------------------------------------------------------
# 4.8 Written Interpretation
# ------------------------------------------------------------

print("\nBivariate Analysis Interpretation")
print("-" * 30)

print(
    "Sex: Female passengers had a substantially higher survival "
    "rate than male passengers, showing a strong relationship "
    "between sex and survival."
)

print(
    "Passenger class: First-class passengers generally had a "
    "higher survival rate than second- and third-class passengers."
)

print(
    "Sex and passenger class together: Survival varied by both "
    "sex and class, with female passengers generally having "
    "higher survival rates and first-class passengers having "
    "an advantage."
)

print(
    "Correlation analysis: The two strongest correlations were "
    "identified by ranking all off-diagonal feature pairs by "
    "the absolute value of their correlation coefficient."
)

print("\n" + "=" * 60)
print("TASK 4 - BIVARIATE ANALYSIS COMPLETED")
print("=" * 60)
# ============================================================
# TASK 5 - MULTIVARIATE ANALYSIS / DATA STORY
# ============================================================

print("\n")
print("=" * 60)
print("TASK 5 - MULTIVARIATE ANALYSIS / DATA STORY")
print("=" * 60)

# ------------------------------------------------------------
# 5.1 Create Age Groups
# ------------------------------------------------------------

print("\nCreating Age Groups")
print("-" * 30)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 50, 65, 100],
    labels=[
        "Child",
        "Teenager",
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior"
    ],
    include_lowest=True
)

print(df["age_group"].value_counts().sort_index())

# ------------------------------------------------------------
# 5.2 Survival Rate by Sex and Passenger Class
# ------------------------------------------------------------

print("\nSurvival Rate by Sex and Passenger Class")
print("-" * 30)

sex_class_survival = (
    df.groupby(
        ["sex", "pclass"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)

sex_class_survival.rename(
    columns={"survived": "survival_rate"},
    inplace=True
)

print(sex_class_survival)

# ------------------------------------------------------------
# 5.3 Survival Rate by Age Group and Sex
# ------------------------------------------------------------

print("\nSurvival Rate by Age Group and Sex")
print("-" * 30)

age_sex_survival = (
    df.groupby(
        ["age_group", "sex"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)

age_sex_survival.rename(
    columns={"survived": "survival_rate"},
    inplace=True
)

print(age_sex_survival)

# ------------------------------------------------------------
# 5.4 Survival Rate by Age Group and Passenger Class
# ------------------------------------------------------------

print("\nSurvival Rate by Age Group and Passenger Class")
print("-" * 30)

age_class_survival = (
    df.groupby(
        ["age_group", "pclass"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)

age_class_survival.rename(
    columns={"survived": "survival_rate"},
    inplace=True
)

print(age_class_survival)

# ------------------------------------------------------------
# 5.5 Multivariate Heatmap
# ------------------------------------------------------------

heatmap_data = (
    df.pivot_table(
        index="age_group",
        columns="pclass",
        values="survived",
        aggfunc="mean",
        observed=True
    )
    .mul(100)
)

plt.figure(figsize=(9, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Titanic Survival Rate by Age Group and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age Group")

plt.tight_layout()

plt.savefig(
    "multivariate_survival_heatmap.png",
    dpi=300
)

plt.close()

print(
    "\nMultivariate heatmap saved as "
    "multivariate_survival_heatmap.png"
)

# ------------------------------------------------------------
# 5.6 Data Story
# ------------------------------------------------------------

print("\nMultivariate Data Story")
print("-" * 30)

highest_sex_class = sex_class_survival.loc[
    sex_class_survival["survival_rate"].idxmax()
]

lowest_sex_class = sex_class_survival.loc[
    sex_class_survival["survival_rate"].idxmin()
]

print(
    f"Highest survival group by sex and class: "
    f"{highest_sex_class['sex']} passengers in "
    f"Class {int(highest_sex_class['pclass'])} "
    f"with a survival rate of "
    f"{highest_sex_class['survival_rate']:.2f}%."
)

print(
    f"Lowest survival group by sex and class: "
    f"{lowest_sex_class['sex']} passengers in "
    f"Class {int(lowest_sex_class['pclass'])} "
    f"with a survival rate of "
    f"{lowest_sex_class['survival_rate']:.2f}%."
)

print(
    "Overall, passenger class and sex were important factors "
    "associated with survival."
)

print(
    "First-class passengers generally had better survival "
    "outcomes than passengers in lower classes."
)

print(
    "Female passengers generally had substantially higher "
    "survival rates than male passengers."
)

print(
    "Age also influenced survival, although its relationship "
    "was weaker than the relationships observed for sex and "
    "passenger class."
)

print("\n" + "=" * 60)
print("TASK 5 - MULTIVARIATE ANALYSIS COMPLETED")
print("=" * 60)


# ============================================================
# TASK 6 - EXPLORATORY STANDARDIZATION
# ============================================================

print("\n")
print("=" * 60)
print("TASK 6 - EXPLORATORY STANDARDIZATION")
print("=" * 60)

# ------------------------------------------------------------
# 6.1 Before Standardization
# ------------------------------------------------------------

print("\nBefore Standardization")
print("-" * 30)

print("\nAge:")
print(f"Mean: {df['age'].mean():.4f}")
print(f"Standard Deviation: {df['age'].std():.4f}")

print("\nFare:")
print(f"Mean: {df['fare'].mean():.4f}")
print(f"Standard Deviation: {df['fare'].std():.4f}")


# ------------------------------------------------------------
# 6.2 Z-Score Standardization
# Formula:
# z = (x - mean) / standard deviation
# ------------------------------------------------------------

df["age_standardized"] = (
    (df["age"] - df["age"].mean())
    / df["age"].std()
)

df["fare_standardized"] = (
    (df["fare"] - df["fare"].mean())
    / df["fare"].std()
)


# ------------------------------------------------------------
# 6.3 After Standardization
# ------------------------------------------------------------

print("\nAfter Standardization")
print("-" * 30)

print("\nAge Standardized:")
print(
    f"Mean: {df['age_standardized'].mean():.6f}"
)
print(
    f"Standard Deviation: {df['age_standardized'].std():.6f}"
)

print("\nFare Standardized:")
print(
    f"Mean: {df['fare_standardized'].mean():.6f}"
)
print(
    f"Standard Deviation: {df['fare_standardized'].std():.6f}"
)


# ------------------------------------------------------------
# 6.4 Before vs After Comparison
# ------------------------------------------------------------

print("\nBefore vs After Comparison")
print("-" * 30)

standardization_summary = pd.DataFrame({
    "Feature": ["age", "fare"],
    
    "Before Mean": [
        df["age"].mean(),
        df["fare"].mean()
    ],
    
    "Before Std": [
        df["age"].std(),
        df["fare"].std()
    ],
    
    "After Mean": [
        df["age_standardized"].mean(),
        df["fare_standardized"].mean()
    ],
    
    "After Std": [
        df["age_standardized"].std(),
        df["fare_standardized"].std()
    ]
})

print(
    standardization_summary.round(6).to_string(index=False)
)


# ------------------------------------------------------------
# 6.5 Interpretation
# ------------------------------------------------------------

print("\nStandardization Interpretation")
print("-" * 30)

print(
    "Age and fare were standardized using the z-score formula "
    "(x - mean) / standard deviation."
)

print(
    "After standardization, both transformed variables have "
    "approximately mean 0 and standard deviation 1."
)

print(
    "This confirms that the exploratory standardization was "
    "performed correctly on the full cleaned DataFrame."
)

print(
    "These standardized columns are used only for EDA sanity "
    "checking and are NOT used in the modeling pipeline."
)


# ------------------------------------------------------------
# 6.6 Optional Visualization - Before vs After
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age
axes[0].hist(
    df["age"],
    bins=30,
    alpha=0.7,
    label="Original Age"
)

axes[0].hist(
    df["age_standardized"],
    bins=30,
    alpha=0.7,
    label="Standardized Age"
)

axes[0].set_title("Age: Before vs After Standardization")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")
axes[0].legend()


# Fare
axes[1].hist(
    df["fare"],
    bins=30,
    alpha=0.7,
    label="Original Fare"
)

axes[1].hist(
    df["fare_standardized"],
    bins=30,
    alpha=0.7,
    label="Standardized Fare"
)

axes[1].set_title("Fare: Before vs After Standardization")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "standardization_before_after.png",
    dpi=300
)

plt.close()

print(
    "\nStandardization comparison chart saved as "
    "standardization_before_after.png"
)

print("\n" + "=" * 60)
print("TASK 6 - STANDARDIZATION COMPLETED")
print("=" * 60)
