import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ============================================================
# QUICK CHART-ONLY VERSION - TASKS 1 TO 6
# ============================================================

print("=" * 60)
print("TASKS 1-6 - CHART-ONLY EDA")
print("=" * 60)

# ------------------------------------------------------------
# Load already-cleaned data
# ------------------------------------------------------------
df = pd.read_csv("titanic_clean.csv")

print("Dataset loaded:", df.shape)

# Create charts folder
os.makedirs("charts", exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# TASK 1 - BASIC DISTRIBUTION CHARTS
# ============================================================

print("\nTASK 1 - BASIC DISTRIBUTION CHARTS")

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("charts/task1_age_distribution.png", dpi=120)
plt.close()

# Fare distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("charts/task1_fare_distribution.png", dpi=120)
plt.close()

print("Task 1 charts saved.")


# ============================================================
# TASK 2 - CATEGORICAL DISTRIBUTION
# ============================================================

print("\nTASK 2 - CATEGORICAL DISTRIBUTION CHARTS")

# Survival count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="survived")
plt.title("Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("charts/task2_survival_count.png", dpi=120)
plt.close()

# Sex count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex")
plt.title("Passenger Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("charts/task2_sex_count.png", dpi=120)
plt.close()

# Passenger class count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="pclass")
plt.title("Passenger Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("charts/task2_pclass_count.png", dpi=120)
plt.close()

print("Task 2 charts saved.")


# ============================================================
# TASK 3 - UNIVARIATE ANALYSIS
# ============================================================

print("\nTASK 3 - UNIVARIATE ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Age histogram
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age - Histogram")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Number of Passengers")

# Age boxplot
sns.boxplot(x=df["age"], ax=axes[0, 1])
axes[0, 1].set_title("Age - Box Plot")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("")

# Fare histogram
sns.histplot(df["fare"], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare - Histogram")
axes[1, 0].set_xlabel("Fare")
axes[1, 0].set_ylabel("Number of Passengers")

# Fare boxplot
sns.boxplot(x=df["fare"], ax=axes[1, 1])
axes[1, 1].set_title("Fare - Box Plot")
axes[1, 1].set_xlabel("Fare")
axes[1, 1].set_ylabel("")

plt.tight_layout()
plt.savefig("charts/task3_univariate_age_fare.png", dpi=120)
plt.close()

print("Task 3 chart saved.")


# ============================================================
# TASK 4 - BIVARIATE ANALYSIS
# ============================================================

print("\nTASK 4 - BIVARIATE ANALYSIS")

# Survival by sex
sex_survival = df.groupby("sex")["survived"].mean().reset_index()
sex_survival["survived"] *= 100

plt.figure(figsize=(7, 5))
sns.barplot(data=sex_survival, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("charts/task4_survival_by_sex.png", dpi=120)
plt.close()

# Survival by passenger class
class_survival = df.groupby("pclass")["survived"].mean().reset_index()
class_survival["survived"] *= 100

plt.figure(figsize=(7, 5))
sns.barplot(data=class_survival, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("charts/task4_survival_by_class.png", dpi=120)
plt.close()

# Correlation heatmap - exactly six required columns
corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/task4_correlation_heatmap.png", dpi=120)
plt.close()

print("Task 4 charts saved.")


# ============================================================
# TASK 5 - MULTIVARIATE ANALYSIS
# ============================================================

print("\nTASK 5 - MULTIVARIATE ANALYSIS")

# ------------------------------------------------------------
# Chart 1 - Survival by Sex and Class
# ------------------------------------------------------------
sex_class = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .reset_index()
)

sex_class["survival_rate"] = sex_class["survived"] * 100

plt.figure(figsize=(9, 6))
sns.barplot(
    data=sex_class,
    x="pclass",
    y="survival_rate",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("charts/task5_survival_sex_class.png", dpi=120)
plt.close()


# ------------------------------------------------------------
# Chart 2 - Age Group and Sex
# ------------------------------------------------------------

bins = [0, 12, 19, 35, 50, 65, 100]
labels = [
    "Child",
    "Teenager",
    "Young Adult",
    "Adult",
    "Middle Age",
    "Senior"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

age_sex = (
    df.groupby(["age_group", "sex"], observed=True)["survived"]
    .mean()
    .reset_index()
)

age_sex["survival_rate"] = age_sex["survived"] * 100

plt.figure(figsize=(11, 6))
sns.barplot(
    data=age_sex,
    x="age_group",
    y="survival_rate",
    hue="sex"
)
plt.title("Survival Rate by Age Group and Sex")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("charts/task5_survival_age_sex.png", dpi=120)
plt.close()


# ------------------------------------------------------------
# Chart 3 - Age Group and Passenger Class
# ------------------------------------------------------------

age_class = (
    df.groupby(["age_group", "pclass"], observed=True)["survived"]
    .mean()
    .reset_index()
)

age_class["survival_rate"] = age_class["survived"] * 100

plt.figure(figsize=(11, 6))
sns.barplot(
    data=age_class,
    x="age_group",
    y="survival_rate",
    hue="pclass"
)
plt.title("Survival Rate by Age Group and Passenger Class")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("charts/task5_survival_age_class.png", dpi=120)
plt.close()


# ------------------------------------------------------------
# Chart 4 - Multivariate Heatmap
# ------------------------------------------------------------

pivot = df.pivot_table(
    values="survived",
    index="sex",
    columns="pclass",
    aggfunc="mean"
) * 100

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100
)
plt.title("Survival Rate Heatmap: Sex vs Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Sex")
plt.tight_layout()
plt.savefig("charts/task5_multivariate_survival_heatmap.png", dpi=120)
plt.close()

print("Task 5 charts saved.")


# ============================================================
# TASK 6 - STANDARDIZATION BEFORE / AFTER
# ============================================================

print("\nTASK 6 - STANDARDIZATION")

features = ["age", "fare"]

before_mean = df[features].mean()
before_std = df[features].std()

scaler = StandardScaler()

standardized = scaler.fit_transform(df[features])

standardized_df = pd.DataFrame(
    standardized,
    columns=["age_standardized", "fare_standardized"]
)

after_mean = standardized_df.mean()
after_std = standardized_df.std()

# ------------------------------------------------------------
# Before vs After chart
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Feature": features,
    "Before Mean": before_mean.values,
    "Before Std": before_std.values,
    "After Mean": after_mean.values,
    "After Std": after_std.values
})

print("\nStandardization Comparison")
print(comparison.round(4))

# Plot means
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    x - width / 2,
    comparison["Before Mean"],
    width,
    label="Before Mean"
)

ax.bar(
    x + width / 2,
    comparison["After Mean"],
    width,
    label="After Mean"
)

ax.set_xlabel("Feature")
ax.set_ylabel("Mean")
ax.set_title("Standardization - Before vs After Mean")
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend()

plt.tight_layout()
plt.savefig(
    "charts/task6_standardization_before_after.png",
    dpi=120
)
plt.close()

# Standard deviation chart
fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    x - width / 2,
    comparison["Before Std"],
    width,
    label="Before Std"
)

ax.bar(
    x + width / 2,
    comparison["After Std"],
    width,
    label="After Std"
)

ax.set_xlabel("Feature")
ax.set_ylabel("Standard Deviation")
ax.set_title("Standardization - Before vs After Standard Deviation")
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend()

plt.tight_layout()
plt.savefig(
    "charts/task6_standardization_std.png",
    dpi=120
)
plt.close()

print("Task 6 charts saved.")


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("TASKS 1-6 CHARTS COMPLETED")
print("=" * 60)

print("\nCharts created inside:")
print("analytics/charts/")