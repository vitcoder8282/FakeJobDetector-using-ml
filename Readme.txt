# 🕵️‍♂️ Fake Job Posting Detection using Machine Learning (NLP + Logistic Regression)

This project is a **Machine Learning-based classifier** that detects **fake job postings** from textual job descriptions using **Natural Language Processing (NLP)** techniques and a **Logistic Regression** model.

---

## 🚀 Project Overview

With the rise of online job platforms, fake job postings have become a serious issue, wasting candidates’ time and risking privacy.  
This project uses **TF-IDF Vectorization** and **Logistic Regression** to identify fake job listings based on text content, job titles, locations, and descriptions.

---

## 📂 Dataset

The dataset used is from Kaggle:  
🔗 [Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

The dataset contains **job descriptions labeled as real (0)** or **fake (1)**.  
Each job posting includes fields like `title`, `location`, `description`, `requirements`, and more.

---

## 🧠 Libraries Used

| Library | Purpose |
|----------|----------|
| **pandas** | Data manipulation and cleaning |
| **numpy** | Numerical computations |
| **matplotlib / seaborn** | Data visualization (heatmaps, plots) |
| **sklearn (scikit-learn)** | Machine learning model, vectorization, and metrics |
| **pickle** | Saving and loading trained model/vectorizer |

---

## ⚙️ Workflow Breakdown

### 1️⃣ Data Loading
- The dataset (`fake_job_postings.csv`) is loaded using pandas.
- The shape and data info are printed to understand structure and missing values.

### 2️⃣ EDA (Exploratory Data Analysis)
- Checked for missing/null values.
- Analyzed distribution of real vs fake job postings.
- Inspected datatypes and column details.

### 3️⃣ Text Preprocessing
- Combined multiple text fields: `title`, `location`, `description`, and `requirements` into one column `text`.
- Handled missing values using `.fillna('')`.
- Converted the `fraudulent` column into integers (0 or 1).

### 4️⃣ NLP Vectorization
- Used **TF-IDF Vectorization** to convert textual data into numerical vectors.
- Considered the top **5000 most frequent words**, ignoring common English stop words.

```python
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df["text"])
y = df["fraudulent"]
