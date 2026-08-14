\# Analytics Module — Titanic EDA and Predictive Modeling



\## Overview



This module implements a complete analyst-to-data-scientist workflow using the classic Titanic dataset.



The workflow covers:



\- Dataset profiling

\- Missing-value analysis and cleaning

\- Univariate analysis

\- Bivariate analysis

\- Multivariate data storytelling

\- Exploratory standardization

\- Stratified train/test splitting

\- Leakage-safe preprocessing

\- Classification using Logistic Regression, Decision Tree, and Random Forest

\- Model evaluation

\- Imbalance handling

\- Random Forest hyperparameter tuning

\- Multivariate linear regression

\- Final model comparison

\- Complete model pipeline saving and reloading



The cleaned Titanic dataset is reused throughout the workflow.



\---



\# Task 1 — Dataset Profiling



The Titanic dataset contains 891 rows and 15 columns before cleaning.



The target variable is `survived`, where:



\- `0` = did not survive

\- `1` = survived



The original dataset contained missing values in:



| Column | Missing Count | Missing % |

|---|---:|---:|

| age | 177 | 19.87% |

| embarked | 2 | 0.22% |

| deck | 688 | 77.22% |

| embark\_town | 2 | 0.22% |



The dataset was saved immediately after loading as `titanic.csv` to provide an offline fallback.



\---



\# Task 2 — Missing Value Handling



The percentage-based threshold rule was applied.



\### Age



`age` had 19.87% missing values.



Because the missing percentage is between 5% and 30%, median imputation was used.



\### Embarked



`embarked` had 0.22% missing values.



Because the missing percentage was below 5%, the affected rows were dropped.



\### Embark Town



`embark\_town` had 0.22% missing values.



Because the missing percentage was below 5%, the affected rows were dropped.



\### Deck



`deck` had approximately 77% missing values.



Because more than 30% of the values were missing, the column was dropped rather than imputed because reliable imputation would not be appropriate.



After cleaning, the dataset contained 889 rows and 14 columns with no remaining missing values.



The cleaned dataset was saved as:



`titanic\_clean.csv`



\---



\# Task 3 — Univariate Analysis



Histograms and box plots were created for both `age` and `fare`.



Generated charts:



\- `age\_histogram.png`

\- `age\_boxplot.png`

\- `fare\_histogram.png`

\- `fare\_boxplot.png`



\## Age Outliers



Using the IQR rule:



\- Q1 = 22.0

\- Q3 = 35.0

\- IQR = 13.0

\- Lower bound = 2.5

\- Upper bound = 54.5

\- Number of outliers = 65



\## Fare Outliers



Using the IQR rule:



\- Q1 = 7.9

\- Q3 = 31.0

\- IQR = 23.1

\- Lower bound = -26.76

\- Upper bound = 65.66

\- Number of outliers = 114



\## Fare Distribution



Fare statistics:



\- Mean = 32.10

\- Median = 14.45

\- Mode = 8.05



Because:



`Mean > Median > Mode`



the fare distribution is right-skewed. A relatively small number of passengers paid very high fares, which pulled the mean upward.



\---



\# Task 4 — Bivariate Analysis



\## Survival Rate by Sex



| Sex | Survival Rate |

|---|---:|

| Female | 74.04% |

| Male | 18.89% |



Female passengers had a substantially higher survival rate than male passengers.



\## Survival Rate by Passenger Class



| Passenger Class | Survival Rate |

|---|---:|

| 1 | 62.62% |

| 2 | 47.28% |

| 3 | 24.24% |



First-class passengers had the highest survival rate, while third-class passengers had the lowest.



\## Survival Rate by Sex and Passenger Class



| Sex | Class | Survival Rate |

|---|---:|---:|

| Female | 1 | 96.74% |

| Female | 2 | 92.11% |

| Female | 3 | 50.00% |

| Male | 1 | 36.89% |

| Male | 2 | 15.74% |

| Male | 3 | 13.54% |



The combination of sex and passenger class shows a much stronger survival pattern than either variable alone.



Female first-class passengers had the highest survival rate at 96.74%, while male third-class passengers had the lowest at 13.54%.



\## Correlation Analysis



The correlation matrix was restricted to exactly these six columns:



\- survived

\- pclass

\- age

\- sibsp

\- parch

\- fare



The boolean-derived columns `adult\_male` and `alone` were excluded.



Generated chart:



`correlation\_heatmap.png`



The two strongest off-diagonal correlations by absolute correlation were:



1\. `pclass` and `fare`: -0.548

2\. `sibsp` and `parch`: 0.415



The negative correlation between passenger class and fare indicates that lower numerical class values, representing higher passenger classes, were generally associated with higher fares.



The positive correlation between `sibsp` and `parch` indicates that passengers traveling with siblings/spouses were also more likely to travel with parents/children.



\---



\# Task 5 — Multivariate Data Story



Generated chart:



`multivariate\_survival\_heatmap.png`



\## Chart 1 — Sex and Passenger Class



Female first-class passengers had the highest survival rate at 96.74%.



Male third-class passengers had the lowest survival rate at 13.54%.



This demonstrates that both sex and passenger class were important factors associated with survival.



\## Chart 2 — Age Group and Sex



Female passengers generally had substantially higher survival rates than male passengers across most age groups.



Children were a notable exception because male and female child survival rates were relatively similar.



This suggests that sex was a stronger survival factor than age for many passenger groups.



\## Chart 3 — Age Group and Passenger Class



Survival generally decreased as passenger class moved from first class to third class.



This pattern was visible across several age groups.



The results suggest that passenger class remained an important factor even when age was considered.



\## Chart 4 — Combined Multivariate Survival Analysis



The combined analysis shows that female first-class passengers were the most likely to survive, while male third-class passengers were the least likely to survive.



Overall, sex and passenger class provide the clearest explanation of survival differences, while age provides additional variation.



\---



\# Task 6 — Exploratory Standardization



`age` and `fare` were standardized using the z-score formula:



`z = (x - mean) / standard deviation`



Before standardization:



| Feature | Mean | Std |

|---|---:|---:|

| Age | 29.3152 | 12.9849 |

| Fare | 32.0967 | 49.6975 |



After standardization:



| Feature | Mean | Std |

|---|---:|---:|

| Age | 0.0000 | 1.0000 |

| Fare | 0.0000 | 1.0000 |



The transformed variables have approximately mean 0 and standard deviation 1.



This confirms that the exploratory standardization was performed correctly.



These standardized columns were used only for EDA and were not used in the modeling pipeline.



Generated chart:



`standardization\_before\_after.png`



\---



\# Task 7 — Stratified Train/Test Split



The target distribution was:



| Class | Count | Percentage |

|---|---:|---:|

| Did not survive | 549 | 61.75% |

| Survived | 340 | 38.25% |



A stratified 80/20 train/test split was used.



Training set:



\- 711 rows

\- 61.74% class 0

\- 38.26% class 1



Testing set:



\- 178 rows

\- 61.80% class 0

\- 38.20% class 1



Stratification is important because the target classes are not perfectly balanced. It ensures that both training and testing sets maintain approximately the same survival/non-survival proportions as the original dataset.



\---



\# Task 8 — Preprocessing Pipeline



The modeling features were:



\- pclass

\- sex

\- age

\- sibsp

\- parch

\- fare

\- embarked



Numeric preprocessing:



1\. Median imputation

2\. StandardScaler



Categorical preprocessing:



1\. Most-frequent imputation

2\. One-hot encoding



The preprocessing was implemented using a `ColumnTransformer`.



All preprocessing steps were fitted only on the training data.



The test data was transformed using the already-fitted preprocessing steps.



This prevents test-set information from leaking into model training.



\---



\# Task 9 — Three Classification Models



Three classifiers were trained on the same stratified train/test split:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest



The Decision Tree was visualized using `plot\_tree` with feature names and class names.



Generated chart:



`decision\_tree.png`



\---



\# Task 10 — Model Evaluation



| Model | Accuracy | Precision | Recall | F1 | AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |

| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |

| Random Forest | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |



Random Forest achieved the highest accuracy, recall, and F1 score.



Logistic Regression achieved the highest AUC.



Generated charts:



\- `logistic\_regression\_confusion\_matrix.png`

\- `decision\_tree\_confusion\_matrix.png`

\- `random\_forest\_confusion\_matrix.png`

\- `roc\_curves.png`



\---



\# Task 11 — Imbalance Handling



Original training distribution:



| Class | Count | Percentage |

|---|---:|---:|

| 0 | 439 | 61.74% |

| 1 | 272 | 38.26% |



Three strategies were compared:



| Strategy | Precision | Recall | F1 |

|---|---:|---:|---:|

| Baseline | 0.7833 | 0.6912 | 0.7344 |

| Class Weight Balanced | 0.7183 | 0.7500 | 0.7338 |

| SMOTE | 0.7353 | 0.7353 | 0.7353 |



SMOTE produced the highest F1 score in this comparison, although the difference from the baseline was very small.



SMOTE was applied only to the training data. The test set remained untouched to prevent data leakage.



\---



\# Task 12 — Random Forest Hyperparameter Tuning



GridSearchCV was used to tune:



\- `n\_estimators`

\- `max\_depth`

\- `max\_features`



Best parameters:



```text

n\_estimators = 200

max\_depth = 15

max\_features = sqrt

