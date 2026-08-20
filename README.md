# Comment Category Prediction — Machine Learning

A machine learning project for classifying online comments into multiple categories using **textual content and numerical metadata**.

The project was developed as part of the **Machine Learning Practice (MLP)** course in the **IIT Madras BS in Data Science and Applications** program.

The final solution is combined predictions from **Logistic Regression and LightGBM** using weighted probability blending.

---

##  Problem Statement

The objective was to predict the category of an unseen online comment using a combination of:

* Comment text
* Engagement information
* Emoticon-related features
* Metadata features

The task involved building a robust multiclass classification pipeline capable of generalizing to unseen test data.

---

##  Dataset

Due to GitHub file size limits, datasets are hosted on Google Drive:

• Train Dataset: https://drive.google.com/file/d/1zKVwYh6xmbKXeKR6t17AyKD5GXrR40Kc/view?usp=sharing       
• Test Dataset: https://drive.google.com/file/d/1ck5ZskXV6ebtAZtqx0rdfWu6mfgXOKlv/view?usp=sharing     
• Sample Submission: https://drive.google.com/file/d/1ND6dK8e5BB1qgpVbk2FcrRX5hPmMGSE-/view?usp=sharing

The training dataset contains **198,000 rows and 15 columns**, while the test dataset contains **102,000 rows and 14 columns**.

### Features

| Feature        | Description                           |
| -------------- | ------------------------------------- |
| `created_date` | Timestamp associated with the comment |
| `post_id`      | Identifier of the associated post     |
| `emoticon_1`   | Emoticon-related numerical feature    |
| `emoticon_2`   | Emoticon-related numerical feature    |
| `emoticon_3`   | Emoticon-related numerical feature    |
| `upvote`       | Number of upvotes                     |
| `downvote`     | Number of downvotes                   |
| `if_1`         | Metadata feature                      |
| `if_2`         | Metadata feature                      |
| `race`         | Categorical metadata                  |
| `religion`     | Categorical metadata                  |
| `gender`       | Categorical metadata                  |
| `disability`   | Boolean metadata                      |
| `comment`      | Textual comment                       |
| `label`        | Target class                          |

The dataset contained substantial missing values in `race`, `religion`, and `gender`, along with one missing comment in the training data.

---

#  Exploratory Data Analysis

The first stage focused on understanding the structure and characteristics of the dataset.

### Key observations

* The dataset contains a mixture of **text, numerical, categorical, and boolean features**.
* `race`, `religion`, and `gender` contained a large proportion of missing values.
* Engagement features such as `upvote` and `downvote` were highly skewed.
* Most comments had zero or very few emoticons.
* The target classes were imbalanced.
* Duplicate comments were also investigated.
* Numerical metadata showed different scales and distributions.

These observations guided the preprocessing and modeling strategy.

---

#  Data Preprocessing

## Text Cleaning

The comment text was normalized before vectorization.

The preprocessing pipeline:

1. Convert text to lowercase.
2. Remove unwanted special characters using regular expressions.
3. Preserve `!` and `?` because they can provide useful emotional or stylistic signals.
4. Normalize whitespace.
5. Handle missing comments safely.

Example:

```python
def clean(text):
    return ' '.join(
        re.sub(r'[^a-z0-9!?\s]', ' ', str(text).lower()).split()
    )
```

---

#  Feature Engineering

The project combines **textual features with numerical metadata** instead of relying only on the comment text.

## 1. Word-level TF-IDF

Word n-grams were used to capture semantic and contextual patterns.

```text
ngram_range = (1, 3)
max_features = 60,000
sublinear_tf = True
min_df = 3
```

This captures:

* Individual words
* Two-word phrases
* Three-word phrases

---

## 2. Character-level TF-IDF

Character n-grams were added to capture sub-word patterns.

```text
ngram_range = (2, 6)
max_features = 40,000
analyzer = "char"
sublinear_tf = True
min_df = 3
```

Character-level features can help capture:

* Misspellings
* Word fragments
* Variations in writing style
* Repeated character patterns
* Sub-word information

---

## 3. Numerical Metadata

The following numerical features were incorporated:

```text
upvote
downvote
emoticon_1
emoticon_2
emoticon_3
if_1
if_2
```

These features provide additional information that may not be directly available from the text.

---

## 4. Sparse Feature Stacking

The word TF-IDF matrix, character TF-IDF matrix, and metadata features were combined using sparse matrices.

```python
X_train = hstack([
    train_word,
    train_char,
    train_meta
]).tocsr()
```

Using CSR matrices helped manage the large high-dimensional feature space efficiently.

---

#  Models Evaluated

Multiple algorithms were experimented with to compare different modeling approaches.

### Models

* Logistic Regression
* Ridge Classifier
* Random Forest
* XGBoost
* LightGBM

This allowed comparison between:

* Linear models
* Bagging-based models
* Gradient boosting models

---

#  Hyperparameter Tuning

`RandomizedSearchCV` was used to explore suitable hyperparameters for Logistic Regression and LightGBM.

### Logistic Regression

Parameters explored included:

* `C`
* `max_iter`

Best parameters:

```text
C = 3.0
max_iter = 1000
```

Best cross-validation macro-F1:

```text
0.783
```

### LightGBM

Parameters explored included:

* `learning_rate`
* `n_estimators`
* `num_leaves`
* `max_depth`

Best parameters from the tuning experiment:

```text
learning_rate = 0.1
n_estimators = 200
num_leaves = 31
max_depth = -1
```

Best cross-validation macro-F1:

```text
0.795
```

---

#  Model Comparison

The models were evaluated using **Macro F1**, which gives equal importance to each class and is useful when dealing with imbalanced multiclass data.

| Model               | Macro F1 |
| ------------------- | -------: |
| LightGBM            |    0.984 |
| Logistic Regression |    0.930 |
| Ridge Classifier    |    0.857 |
| Random Forest       |    0.573 |

XGBoost was also experimented with and achieved a strong overall training performance, but its recall for one minority class was comparatively weaker.

> **Note:** These model-level scores are calculated on the training data in the notebook and should not be interpreted as the Kaggle leaderboard score. The final external benchmark for the project was the Kaggle score of **0.83703**.

---

#  Final Prediction Strategy

Instead of relying on a single model, the final solution blended the probability outputs of Logistic Regression and LightGBM.

```python
weighted_probs = (
    p1_lr * 0.40 +
    p2_lgb * 0.60
)

final_preds = np.argmax(weighted_probs, axis=1)
```

### Final blending weights

```text
Logistic Regression → 40%
LightGBM            → 60%
```

The final predictions were then written into the required Kaggle submission format.

---

#  Experiments & Iterations

Several approaches were tested during development.

### Ensemble experiments

I experimented with:

* Logistic Regression
* SGD Classifier with ElasticNet
* LightGBM
* Soft voting
* Hard voting
* Multi-model probability blending

Some ensemble approaches were computationally expensive because of the large sparse feature matrices and exceeded available memory.

Reducing the feature space produced approximately **0.829** Kaggle performance, while some hard-voting experiments performed substantially worse.

These experiments helped identify a simpler and more effective **Logistic Regression + LightGBM weighted blend**.

---

#  Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* SciPy

### Machine Learning

* Scikit-learn
* LightGBM
* XGBoost

### NLP

* TF-IDF
* Word n-grams
* Character n-grams
* Regex-based text preprocessing

### Visualization

* Matplotlib
* Seaborn

### Development / Competition

* Jupyter Notebook
* Kaggle
* GitHub

---

#  Project Structure

```text
comment-category-prediction/
│
├── notebook/
│   └── comment_category_prediction.ipynb
│
├── README.md
│
└── submission.csv
```

---

#  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd comment-category-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn lightgbm xgboost
```

### 3. Open the notebook

```bash
jupyter notebook
```

Run the notebook from top to bottom after placing the competition dataset in the expected input directory.

---

#  Key Takeaways

This project provided hands-on experience with an end-to-end machine learning workflow:

**Data → EDA → Cleaning → Feature Engineering → TF-IDF → Model Training → Hyperparameter Tuning → Model Comparison → Probability Blending → Kaggle Submission**

The most important learning outcomes were:

* Handling large mixed-type datasets
* Working with high-dimensional sparse text features
* Combining NLP features with structured metadata
* Handling class imbalance
* Comparing multiple machine learning algorithms
* Performing hyperparameter tuning
* Experimenting with ensemble strategies
* Optimizing memory usage for large sparse matrices
* Evaluating models on unseen Kaggle data

---

##  Achievement

**Kaggle Score: 0.83703 | Rank: 80 out of 2744 | S Grade (10 GPA)**

Built as part of the **Machine Learning Practice (MLP)** coursework at **IIT Madras**.
